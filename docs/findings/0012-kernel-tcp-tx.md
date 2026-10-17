# 0012: kernel TCP TX correctness control

Observed 2026-10-10 UTC on one RTX 5090, CUDA 13, Linux
7.3.0-070300rc3-generic, GCC 15.2.0, PyTorch 2.14.1+cu130,
Transformers 5.19.0. This is an implementation/correctness increment, not a
paper performance result or the complete resident-GPU serving path.

## Defects and ownership repair

The previous prototype's TCP client timed out after Qwen wrote 16 tokens into
the page. Two abrupt host reboots occurred during that earlier work; their
cause remains unknown. No successful kernel TX was proven by those runs.

The previous init code treated a bpffs file's private_data as a BPF map,
closed the file, then incremented that wrong object. Linux
[v7.3-rc3 bpffs source](https://github.com/torvalds/linux/blob/v7.3-rc3/kernel/bpf/inode.c)
shows private_data is a seq_file. The new loader uses bpf_obj_get and passes
its actual BPF FD to finit_module in the same process. bpf_map_get validates
and references it; type, mmap flag, entry count and canonical value size are
checked before accessing the array. The logged map ID agrees with ingress.
There is no filp_open/private_data cast or dangling map reference.

[Linux TCP lookup](https://github.com/torvalds/linux/blob/v7.3-rc3/net/ipv4/inet_hashtables.c)
already returns a reference. TX retains that reference until the slot ends,
uses the loading process netns and sends via tcp_sendmsg_locked outside RCU,
under the socket lock. It never dereferences the concurrently detachable
sk_socket. Every positive return advances a byte offset, including a partial
token. Transient errors are retried. Disconnect cannot recycle PENDING;
the executor's DONE publication releases ownership after its final write.
The module derives offsets and types from the canonical queue header.

## Reproduce

Build the normal probes/tools and the optional module against matching headers:

```sh
make probes tools
make -C module CC=gcc-15
PYTHONPATH=executor python3 tests/kernel_tx.py
PYTHONPATH=executor python3 tests/qwen_kernel_tx.py
```

Root privileges are needed. These tests own the bpf-gpu-server tc hook in their
network namespace, start only their own executor, unload their test module and
detach on exit. The protocol test's finit_module syscall is for Linux x86-64.
For a manual host-driven Qwen control: attach as usual, run
`./build/bpfusion_load tx-load`, then launch the existing executor with
`--tcp --no-send`. Stop that executor, `rmmod bfusion_tx`, and detach afterwards.

## Observed results

Raw logs:

- [kernel_tx_protocol_20261010.txt](../../bench/results/kernel_tx_protocol_20261010.txt)
- [qwen_kernel_tx_20261010.txt](../../bench/results/qwen_kernel_tx_20261010.txt)

The protocol test rejects the incompatible stats map, verifies 24 byte-exact
TCP replies and two reset clients, cycles the eight-slot ring 26 times, and
holds PENDING for 80 ms after output publication before DONE. No premature
recycle occurs. Single-byte client reads validate stream framing on receipt;
a kernel partial send of a non-multiple-of-four length was not forced by this
run. That case is handled by exact byte accounting in the implementation.
The module reports 5,892 bytes, 24 completions and two abandoned requests.

The real model control uses Qwen/Qwen2.5-0.5B-Instruct, fp16 greedy batch-1 HF
in both modes. Four prompts, output lengths 1/16/64 and two repetitions give
24 replies per mode. Every reply in both userspace-send and kernel-TX modes
matches every token ID from an independent same-HF greedy oracle. Both modes
completed without failed requests. The same generation algorithm and dtype
avoid a cross-engine confound. The oracle model is released before serving.
Cold starts, fixed mode ordering and the small sequential correctness workload
make the printed TTFT/TPOT descriptive diagnostics; no speedup or tail-latency
claim is made.

Both test processes ended, the module was unloaded, GPU allocation returned
to 15 MiB, and the host boot ID remained unchanged across these attempts.
This does not identify or exclude the earlier reboot cause.

## Remaining scope

No userspace token send occurs in the kernel control, but Python still accepts
connections and drives every prefill/decode operation. The module polls the
page; it is not completion-event-driven. Ingress expects one complete binary
request in a TCP segment; framing, retransmission deduplication, multiple
producers, HTTP, GPU tokenization and a resident real-model kernel remain
open. The full no-steady-userspace-worker goal remains unchanged.
