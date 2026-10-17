# Cross-node TCP/HTTP with reused model graphs

## Question and current status

Does removing the steady userspace request worker improve client latency or
throughput over efficient native networking on a real cross-node path,with
identical upstream GPU computation?

The [plan](../../bench/results/physical_network_plan_20261011.md) and
[fresh plan review](../../bench/results/physical_network_plan_review_20261011.md)
admit a supporting comparison. There is no formal paper RQ or external SOTA
claim. Finding0028 remains separately closed and inconclusive.
Address compatibility and all four real cross-node preflight executors pass.
The80-cell full comparison is complete. Its primary paired latency/throughput
intervals contain zero for both models; reliable gains remain unproven.
Fresh independent result review is recorded with the full-run evidence below.

## Necessary source adaptation

Graph executors reuse existing BF_BIND with127.0.0.1 default. Kernel TX reads
the concrete local IPv4 address from its owned listener for established-socket
lookup. This preserves network-order address/peer-port and host-order local
port semantics. Legacy reply_addr remains a fallback without a listener FD.
Kernel listener validation includes family,protocol,state,port,address and
network namespace. Wildcard listener,IPv6 and VRF are not qualified.

The four-source change is24 inserted/12 deleted lines: graph binding,kernel
lookup authority,and destination substitutions in existing test/benchmark.
Model computation,precision,eight graph/KV banks,dispatcher,queue ABI,TX
coalescing and wait policies are unchanged.
[Build receipt](../../bench/results/physical_network_build_20261011.stdout.txt)
contains four successful graph builds and the owning module Kbuild.
Existing compiler/pahole warnings and unavailable-vmlinux BTF skip are retained.

## Dependency qualification and default compatibility

[Own-Pod-IP strict stdout](../../bench/results/physical_network_binding_qualification_20261011.stdout.txt)
records normal exits for both kernel models and Qwen timed networking:
1005 valid requests,61296 exact independent eager-HF tokens,777 rejections,
plus5000 empty connections per invocation. Fragmentation,reuse/wrap,
pipeline/half-close,mixed lengths,invalid IDs,after-ready network tracing
and cleanup pass. These connects originate inside the same Workspace;
they prove binding/lookup,not physical delivery.

[Default-loopback preflight raw](../../bench/results/physical_network_loopback_preflight_20261011.jsonl)
and [stdout/native analysis](../../bench/results/physical_network_loopback_preflight_20261011.stdout.txt)
complete three modes,Qwen,12 exact measured one-token replies,three normal
executors,one hostlaunch/264modelgraphsteps/eightbanks/zeroerrors each.
The busy row is compatibility dependency evidence,not another full baseline.

## Actual cross-node preflight

The original server remains bpfusion-dev on lab,PodIP10.42.1.238.
A disposable benchmark client Job runs on raspberrypi,PodIP10.42.0.118,
with Python3.12.12-slim-bookworm and no model or project PVC.
[Exact temporary Job](../../bench/results/physical_network_client_job_20261011.json)
and [environment receipts](../../bench/results/physical_network_environment_20261011.txt)
retain actual identities,imageID,placement and cleanup contract.

[Smol raw](../../bench/results/physical_network_smol_preflight_20261011.jsonl),
[Qwen raw](../../bench/results/physical_network_qwen_preflight_20261011.jsonl)
and [both direct analyses](../../bench/results/physical_network_preflight_analysis_20261011.stdout.txt)
complete four normal executors,four cells,16 measured exact tokens.
Eight warmups/executor are separate and also checked against independent
references. Each executor reports1hostlaunch,264modelgraphsteps,banks8,
graph_error0,maxactive1,as expected for this preflight. Timed controls engage
epoll_pwait2 60000ns,published12/completed12/rejected0 with kernel TX absent.
Each resident kernel lifecycle reports2000bytes,12completions,zeroabandoned/
retries,2accepted/closed.

The [native client adapter](../../bench/results/physical_network_client_20261011.py)
reuses request()/connect() verbatim; [literal invocations](../../bench/results/physical_network_preflight_invocations_20261011.txt)
retain actual case/cell calls. [Analysis](../../bench/results/physical_network_analysis_20261011.py)
recomputes client metrics without fabricated server CPU.
Client-local monotonic durations require no cross-machine clock subtraction.
Descriptive quantiles use nearest rank,unlike older interpolated tables.

[Inner TCP capture](../../bench/results/physical_network_preflight_tcp_20261011.pcap)
and [decoded capture/log](../../bench/results/physical_network_preflight_tcp_20261011.txt)
record the actual10.42.0.118↔10.42.1.238:39403 tuples across flannel/CNI:
3708 captured/received records,zero kernel drops. Multiple forwarding
interfaces duplicate records; this is not3708 unique packets or requests.
Snapshot length256 truncates some payloads; client oracle is the complete
output check. Interface labels are read-time labels; retain SLL2 indices.

[Carrier header examples](../../bench/results/physical_network_preflight_carrier_20261011.txt)
and route/physical-device receipt identify enp129s0 and
128.114.59.130↔128.114.59.42 UDP41641 Tailscale transport.
That capture also includes unrelated encrypted peer traffic; retain bounded
headers only and attribute no carrier byte rate or energy to bpf-gpu-server.
This is cross-node Linux TCP over existing flannel/Tailscale and a physical
Ethernet underlay,not bare-NIC TCP,GPUDirect or an OS-bypass service.
Both preflight captures are stopped before full collection.

## Complete matched cross-node comparison

Frozen source is eaa9c8e6bd5f79d35c0a260770833c7071302380. Collection on
2026-10-12 completes all80 planned cells,5120 measured requests,166400 exact
measured tokens and20 normal executors,without failed/excluded/retried rows.
Each model contributes40 cells,2560 requests,83200 tokens and10 executors.
Eight checked warmups per executor are additional and excluded from these counts.

[Smol full raw](../../bench/results/physical_network_smol_20261011.jsonl),
[Qwen full raw](../../bench/results/physical_network_qwen_20261011.jsonl),
[actual client footers/server wrappers](../../bench/results/physical_network_full_invocations_20261011.txt)
and [full analysis](../../bench/results/physical_network_full_analysis_20261011.stdout.txt)
retain actual outputs and recompute all client timings,TPS,coverage and oracle
agreement. [Fresh independent result review](../../bench/results/physical_network_result_review_20261011.md)
audits the complete comparison.

| Model | Metric/condition | Kernel resident | Efficient userspace | Difference favoring kernel | Nominal paired t95,df4 |
|---|---|---:|---:|---:|---|
| SmolLM2-135M | Mean TTFT ms,gen1/c1 |3.804418|3.590233|-0.214185|[-0.526943,0.098572]|
| Qwen2.5-0.5B | Mean TTFT ms,gen1/c1 |3.189486|3.034464|-0.155022|[-0.564566,0.254522]|
| SmolLM2-135M | Output tokens/s,gen64/c8 |1825.315649|1843.902505|-18.586856|[-70.530540,33.356828]|
| Qwen2.5-0.5B | Output tokens/s,gen64/c8 |1420.733955|1418.596725|2.137230|[-4.588850,8.863310]|

TTFT difference is userspace-minus-kernel; throughput is kernel-minus-userspace.
Every primary interval contains zero: no reliable advantage and no equivalence
claim. Smol pair3's throughput difference of-93.256038tokens/s remains in the
five-pair estimate; no verified cause or selected exclusion is assigned.
Other cells and nearest-rank quantiles are descriptive in the complete analysis.
There is no familywide superiority test or server/total CPU measurement.

All20 executor logs report one hostlaunch,8580modelgraphsteps,eight banks,
maxactive8 and zero graph errors. Each timed alternative reports epoll_pwait2
60000ns,264published/264completed/zero rejected,with kernel TX absent.
All ten resident journal lifecycles report55176bytes,264completed,
zeroabandoned/retries,19accepted/closed. Reserved mask is zero and all eight
slots are FREE in the archived resident stats.

[Full environment/restoration receipt](../../bench/results/physical_network_full_environment_20261011.txt)
retains the explicit UTC journal window and API identities. That retained
window has no matching listed kernel/GPU/memory/filesystem fault lines.
Initial and per-executor GPU observations are discrete; early query return
statuses were not retained. After-run software versions are observations at
that time,not a continuous full-window version or GPU-isolation audit.
No host/software/kernel/model-compute change is made for this collection.

At2026-10-12T17:41:39Z,the temporary client Job and Pod are absent; the original
server UID remains Running/Ready with zero restarts. Own model processes,
kernel module and BPF pins are absent,with a successful final GPU query showing
15MiB and no compute applications. Unrelated tools/setup_env.sh is preserved.
The benchmark resource contract is restored.

## Boundary and next action

One management SSH attempt times out across HA servers; later real SSH and
owning API reads confirm Ready nodes and no client Job. No reboot,route or
credential change is made. A later local-path/remote-kubectl setup error
creates no object; stdin succeeds. These are retained setup observations,
not failed model samples. A journal command with ambiguous local timestamps
fails before saving data; the corrected explicit UTC window is retained.
That window has normal own module lifecycles and no matching listed kernel/
GPU/memory/filesystem fault lines. It does not prove continuous isolation.

Close this completed supporting matrix with its inconclusive outcome and
restored resources. Do not rerun unchanged conditions to obtain a favorable
estimate. The next experiment needs a distinct causal discriminator or a
stronger serving comparison while continuing to reuse backend computation. No total service CPU,full text/
JSON,online batching,arbitrary model support or external serving superiority
is established here; those remain under the full retained goal.
