# Finding 0001 — ingress → userspace wakeup on this kernel

Status: measured on 2026-10-08, `bench/results/ingress_20261008T003403Z.txt`.

## Question

To drive a resident GPU executor from an eBPF request path, the kernel has to be
able to *wake a userspace thread* when a request is parsed. What does the only
programmable mechanism available today — a pollable BPF map — actually cost on
this host, and how does it compare with a kernel module's direct
`wake_up()`? Is the gap small enough that the "no control plane in the steady
state, but userspace runs the GPU" design is still worth building?

## Setup

- Socket filter (`SO_ATTACH_BPF`) on a loopback UDP socket records
  `(payload_timestamp, bpf_ktime_get_ns())` per packet into a
  `BPF_MAP_TYPE_RINGBUF` (`bpf/wake_probe.bpf.c`).
- The sender stamps `CLOCK_MONOTONIC` **inside the payload**, so pairing of
  send/record/wake is exact under loss, reordering or unread datagrams. The
  filter reads the payload at offset 8 (`bpf_skb_load_bytes`), which on this
  kernel is where the UDP header ends; offset 0 is the UDP header, not the
  payload. Recorded `load_ret=0` confirms the read.
- Two consumers: `epoll_wait()` on the ring-buffer map fd (the ordinary way to
  sleep on a pollable map) and a busy-wait `ring_buffer__consume()` spinner.
- Baseline: blocking `recvfrom()` on the same socket, timestamped on return.
- 20 000 packets per run, 50 µs spacing between sends, 3 repetitions.

## Results

| chain | p50 | p90 | p99 | max |
|---|---|---|---|---|
| sendto → BPF socket filter | 1.6–1.7 µs | 1.9 µs | 3.8–7.4 µs | 20–38 µs |
| BPF record → userspace, **busy-wait** | 110–124 ns | 132–168 ns | 204–229 ns | up to 170 µs |
| BPF record → userspace, **epoll_wait** | 2.4–2.5 µs | 5.2–5.4 µs | 135–149 µs | 422–877 µs |
| sendto → blocking `recvfrom()` (baseline) | 3.4–3.5 µs | 6.2–7.7 µs | 142–148 µs | 262 µs–1.6 ms |

Interpretation:

1. The BPF-record → userspace hop with a spinning consumer is ~150 ns p50
   (mean 140–220 ns). Once the consumer is running, the pollable map is a
   shared-memory handoff, not a syscall: `ring_buffer__consume()` reads the
   same pages the kernel wrote. This is the floor for a *userspace* executor
   woken by a pollable map.
2. `epoll_wait()` adds ~11–18× at p50 (2.4 µs vs 150 ns) because every record
   that arrives while the consumer is sleeping must wake a task. At 20 k
   packets / 50 µs spacing, one wakeup per packet is priced in; this is the
   realistic cost of the "sleep until a request lands" policy.
3. A full `sendto → recvfrom()` on loopback is ~3.4 µs p50 — the same order as
   the epoll path, so the ring buffer does not make the datapath cheaper, only
   more programmable (the filter can classify, and — see below — decide where
   the payload is dispatched).
4. The p99/max values of the *blocking* variants (135–150 µs) are
   placement/scheduling outliers, not queueing: the p99 of the spin variant is
   229 ns in the same window, and the busy-wait sender chain keeps a stable p50
   throughout (sender and consumer were not CPU-pinned during these runs).

## Why the busy-wait variant is a *measurement*, not a production path

A spinning consumer used 100 % of a core, and during the first attempt the main
thread's `nanosleep()` starved the spinner outright (100 µs–2 s wakeups). The
numbers are only trustworthy *after* the sampling loop was forced onto
`now_ns()` deadlines with the spinner owning the ring buffer.

## Consequence for the bpf-gpu-server design

A kernel module calling `wake_up()` on a wait-queue would cut the 2.4 µs p50 of
the epoll path to roughly a task-scheduler wakeup. That cost has **not been
measured on this host** (no module was written); the comparison is an
*inference* from first principles, not evidence. A module or a new
device/completion kernel interface is also **not** "abandoning eBPF verifier
programmability" — the target design already expects a new device/completion
interface, with eBPF handling the request path.

Until such an interface exists, the only programmable handoff is the pollable
map, whose cost is ~2.4 µs per wakeup, and only the *first* packet after an
idle period pays it — a consumer that drains a batch per wake pays it once per
batch, not once per request.

**Interim decision (labelled as such):** the current implementation hands the
batch to a *userspace* thread. That makes it a **host-driven intermediate
experiment**, not the final architecture: in the target path userspace only
bootstraps/loads/inits/controls/telemetry, and there is no userspace request
worker in the steady state. The userspace stage exists to get a real
packet→GPU→packet number today and is not evidence that a CPU request worker
is the target.

## Pinning control (2026-10-08, `bench/results/ingress_20261008T075650Z.txt`)

The p99 attribution above was an inference because no pinned run existed.
`tools/wake_probe` now takes a fifth argument: `1` pins the consumer and sender
to CPUs 2/3 with `SCHED_FIFO` priority 90 (`pin_fifo()`). Re-running the same
three variants pinned:

| chain | p50 | p90 | p99 | max |
|---|---|---|---|---|
| sendto → BPF record | 1.0–1.1 µs | 1.4 µs | 1.9–2.0 µs | 24–31 µs |
| BPF record → userspace, **busy-wait** | 94–101 ns | 128–139 ns | 181–186 ns | 24 µs–204 µs |
| BPF record → userspace, **epoll_wait** | 2.0–2.2 µs | 2.7–2.9 µs | **3.7–4.1 µs** | 62–282 µs |
| sendto → blocking `recvfrom()` | 2.9–3.0 µs | 3.8–4.0 µs | 4.7–599 µs | 147 µs–1.2 ms |

This confirms the inference directly: the unpinned epoll p99 (135–149 µs) is a
**scheduling/placement outlier, not queueing** — under pinning it collapses to
3.7–4.1 µs, a 35–40× reduction, matching the ~2 µs p50 plus a bounded wake
latency. The spin variant's p99 is essentially unchanged (204–229 ns unpinned vs
181–186 ns pinned) because it never sleeps, which is exactly why it is the floor.
The `recvfrom` p99 is noisier (one rep still hit 599 µs even pinned); that path
uses the same task-wake machinery as epoll and inherits its sensitivity to
`nanosleep`-driven sender jitter.

**Consequence:** with the consumer pinned, a per-batch `epoll_wait()` wake costs
~2 µs p50 / ~4 µs p99 on this host, and the pollable-map handoff is a
shared-memory read at ~0.1 µs. Neither is a blocker for the design; the
unpinned p99 (which the earlier table reported) should not be quoted as the
cost of the mechanism.

## Open items

- The kernel-module `wake_up()` figure is still unmeasured; the module-free
  comparison (spin vs epoll vs recvfrom) is now pinned-controlled.
- Loopback only: this measures the whole stack above the NIC, so the NIC RX
  contribution is *not* included. A real-NIC measurement is still owed.


