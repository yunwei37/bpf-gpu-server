# Paper artifact

This artifact audits the four-page report in [paper/](paper/README.md), entry
revision ad7982f; the paper owner renamed the main file in06d2c598 and
corrected rounding/claim scope in1e2303ee and5b7dbb75.
The replay discovers the main TeX by its table label rather than its name.
It does not edit paper prose/PDF. Paths are relative to the
source root; programs locate that root from their own file, so the directory
and remote repository may be renamed.

Current source/command names follow the [rename mapping](README.md). Archived
raw filenames, embedded source snapshots and historical finding commands retain
their original names. The offline replay uses those files directly; do not
rename archived inputs or substitute renamed source into a historical run.

## One-command offline replay

Prepare Python3.12 and plotting dependencies once while online:

~~~sh
python3 -m venv ../artifact-env
../artifact-env/bin/python -m pip install -r bench/artifact-requirements.txt
~~~

If ensurepip is absent, use venv --without-pip and an existing pip's
--python option to install into the new environment only.
Tested: Python3.12.3, Matplotlib3.11.2, NumPy2.5.4; no inference packages.

From the source root, after dependency installation:

~~~sh
../artifact-env/bin/python bench/reproduce_paper.py --output ../artifact-output
~~~

The output directory must be new and outside the source tree. The program
rejects socket connect/bind/name-resolution. Original analyzers replay raw
requests, worker logs, CPU counters, output checks and perf stacks; all90 main
graph cells are reconstructed from native binary records. Saved per-cell
metrics are checked against new computations, not used to compute them.

Outputs: comparison.md/json (paper, recomputed value, tolerance and status);
figure2_cpu.png/pdf and figure3_perf.png/pdf plus figure_data.json; raw-recomputed
stage JSONLs; inputs.txt (exact relative read set); interpretation_limits.json.
Figures use unrounded data; their layout differs from pgfplots.

Exit0 means numerical agreement. Exit1 with a completed comparison.json means
a reported discrepancy; an exception means a validation/program failure.
The paper owner corrected the previously reported rounding discrepancy; its
historical receipt is retained below. Table shape changes fail visibly. Allow4–10min. No network, GPU, model weights or original
Workspace is required. Full Git tree:about4.24GB before archive compression.

Current paper5b7dbb75: [full raw replay](bench/results/paper_artifact_20261017/author-update/README.md)
completes136PASS/0MISMATCH/0PAPER_CHANGED, exit0 in214.085s. The previous
rounding report is retained as historical evidence.
The [post-rename replay](bench/results/rename_20261017/README.md) also passes
all136checks, with the same7182inputs and identical figure data. Historical
raw filenames are unchanged by the namespace migration.

## Complete paper-to-data map

R below abbreviates bench/results/. Glob notation denotes committed families.
The unified command above covers every numerical row below. Quantiles are
linear; means are over five process-block statistics, not pooled P99s.
Perf is the explicit single-pair exception.

| Paper item / key numbers | Finding | Raw paths under R | Recalculation |
|---|---|---|---|
| Table1 top:K16/N16/official;11645/11645/12245token/s;3304/3314/611ms;CPU .835/.812/8.545ms | [0078](docs/findings/0078-sustained-official-vllm.md) | official_sustained_20261016/official_remote_full_retry2/ clients, worker JSONLs, resources and server/warmup/exit receipts | official_remote_analysis.py in experiment root |
| Table1 bottom:N16/N32/official;11640/12265/12249token/s;3337/217/406ms;CPU .751/.523/8.327ms | [0080](docs/findings/0080-resident-queue-relief.md) | resident_queue_relief_20261017/queue_relief_full/ same raw families | queue_relief_analysis.py |
| Table2:three CPU/P99 ratios and all95% interval endpoints | [0081](docs/findings/0081-small-graph-network-cpu.md) | graph_network_20261017/confirmation/confirm_s*_p*_{kernel,native}.* including binary records/timers, schedules, patterns and counters | bench/graph_network_analysis.py plus paired log ratios,t95(df4) |
| Figure2:FFT/GEMM user/system splits;37.56/37.96/37.29/38.01us totals | 0081 | same confirmation resource records:process_group stat and cpu_ns | tick deltas and high-resolution CPU separately; figure2_cpu.* |
| Figure3:K26.7/43.8/27.0/2.4%,N84.5/0/0/15.5% | [0073](docs/findings/0073-kernel-cpu-localization.md) | kernel_cpu_profile_20261016/cpu_profile_full/*.perf_script.txt.gz; matching perf.data.gz/header/decode/terminal retained | profile_summary.py ancestor categories; figure3_perf.* |
| Table3 NODELAY:40.5→2.3ms;8842→11260token/s | [0070](docs/findings/0070-tcp-nodelay.md) | tcp_nodelay_20261016/tcp_full/full.jsonl.gz | tcp_analysis.py,closed16/kernel before-after |
| Table3 EOF:4157→0 failures/57600;7793→11623token/s | [0072](docs/findings/0072-connection-reclamation.md) | connection_reclamation_20261016/verified_reclaim_full/ clients/resources/receipts | reclaim_analysis.py |
| Table3 active wait:1.70→.63ms CPU | [0074](docs/findings/0074-active-admission-wait.md) | active_admission_wait_20261016/active_wait_full/ | active_wait_analysis.py |
| Table3 mixed scheduling:1.43→1.23ms,−14% | [0065](docs/findings/0065-mixed-prefill-decode.md) | mixed_batching_20261015/full_qwen.jsonl plus matching GPU/reference/fresh-metadata JSONLs | analyze.py,Qwen96/s,equally BGS_TRACE |
| Table3 slots:3337→217ms | 0080 | Table1 bottom | shared capacity,not eBPF gain |
| Abstract >3.4M final requests;graph3.12M;screen2.52M/60cells;confirmation600000 | 0078,0080,0081 | final matrices above plus graph screen/ original records | replay counts3,465,600,excluding warmups/retired prefixes |
| Approx15× slot tail/17× NODELAY tail | 0080,0070 | corresponding paired matrices | exact15.354/17.689;approximate prose tolerance1;tables half printed unit |
| Transport−10ms/[−64,44];earlier18.7% CPU reduction | 0078,0074 | corresponding raw matrices | paired block contrasts |
| Queue TTFT235→3003ms;generation≈86ms | 0078/0080 diagnosis | original0078 native request timestamps | first/last10s original scheduled-arrival buckets |
| Capacity costs .033ms TPOT/.34J device;10–16× group-CPU gap | 0080,0078 | raw requests/resource windows | separate process/device scopes |
| Screen best CPU reduction5.8%;one-CPU P99≈2.6×;host ratios .990/.925/.988 | 0081 | all60 screen/30 confirmation cells | full-grid and selected-pair replay |
| 11.3+26.2 vs2.8+35.2us;FFT/GEMM totals +1.1/+1.9%;≈38us/request;observer2–34us | 0081 | original process/resource windows | coarse ticks,high-resolution CPU and observer extrema |
| 4-byte send,36-byte release,≈41ms ACK wait | 0070 | tcp_nodelay_20261016/tcp_preflight/*.tcp.jsonl.gz | tcp_diagnostic_analysis.py,longest kernel-before packet episode |
| 2947 CLOSE_WAIT/E2BIG/4157 missing completions | 0072 | connection_reclamation_20261016/errno/kernel_snapshot_now.json and errno logs;full matrix | owned-inode/TCP state08 intersection plus failure count |
| Perf71% coordination/maps,native85% network,10086/2455 samples,4.1×,499Hz | 0073 | raw callchains/header/record command | mutually exclusive observed ancestors,not total CPU |
| Working TCP/HTTP,both models,unchanged cuBLAS/cuFFT,cancellation | [0079](docs/findings/0079-kernel-network-source-integration.md),[0067](docs/findings/0067-opaque-graph-backend.md) | kernel_network_integration_20261017/ and graph qualification linked by0067 | finite qualification;live smoke below |
| No predeclared useful graph CPU/P99 advantage | 0081 | plan.md,frozen selection,full raw matrices | evaluates .90/1.05 bounds,.98 throughput and failures |

Standalone analyzers take their raw directory, except tcp_analysis.py and
mixed analyze.py take their main JSONL path (TCP reads a .gz fallback).
Example, from source root:

~~~sh
python3 bench/results/kernel_cpu_profile_20261016/profile_summary.py \
  bench/results/kernel_cpu_profile_20261016/cpu_profile_full
~~~

Figure1 is a source schematic, not a data plot. Its claims map to
bpf/server.bpf.c, bpf/kernel_tcp.bpf.c, executor/kernel_tcp.h,
executor/vllm_mixed.cu, executor/graph_server.cu and the
[architecture contract](docs/architecture.md). Timer60000ns,24-byte framing
and64-word maximum are code/ABI facts. Workload metadata is checked by replay.
External citation/related-work claims are outside numerical replay.

## Discrepancy and author-facing limits

The initial paper printed Figure3 kernel “other” as2.5%. Raw scheduler103
plus unknown-leaf144 give247/10086×100=2.4489391235%, directly rounded to2.4%.
The other three printed percentages sum97.5%, making2.5% their rounded
complement. No samples are missing. The initial134PASS/1MISMATCH receipt is
historical:the paper owner corrected this value to2.4% in1e2303ee. Generated
figures continue to use unrounded data. The current two-workload CPU prose
from5b7dbb75 is checked against unrounded FFT/GEMM means separately.

Numerical agreement cannot resolve these interpretation issues:

- Crossing0/1 does not prove “identical” throughput or “unchanged” whole-host CPU.
- The32-slot full performance arm is native only. Shared architecture does not
  establish an equal measured kernel32 gain.
- Figure3 is one ordered pair, an exception to the blanket five-block sentence.
  Unknown ancestors occur in9404/10086 kernel samples; network/softirq may
  include unrelated traffic.
- The pre-fix LLM profile does not quantitatively attribute final graph CPU to
  TCP. “Most” of the official-vLLM CPU gap coming from scheduling/protocol
  is an inference. Official used tokens-only HTTP/SSE:both paths return IDs,
  although framing/parsing/scheduler/KV still differ.
- The former “within about1%” CPU sentence was corrected by the paper owner
  in5b7dbb75 to FFT1.1% and GEMM1.9%, matching high-resolution changes of
  +1.06%/+1.93%. The Table3 native-only32-slot footnote was added in1e2303ee;
  these corrections do not remove the other interpretation limits above.
- Exact replay qualifies finite traced cases; ordinary LLM trajectories retain
  finite-reference disagreement flags. Protocol success is not exact parity.
- Service affinity does not create exclusive CPU scarcity; IRQ/foreign work
  remained unconstrained.
- Linux7.3-rc3 is the tested release candidate; broader/stable/fresh-OS/TLS/
  arbitrary-plugin compatibility is not demonstrated.

This thread does not edit paper/; the updates above are paper-owner commits.

## Data availability

All numerical inputs are ordinary committed Git files. inputs.txt records the
actual read set; the [validation receipt](bench/results/paper_artifact_20261017/README.md)
records clean download/source export, new Python environment, commands,
durations, exits and public availability. Complete decoded callchains accompany
binary perf recordings. Fresh symbolization can require historical kernel/BPF
symbols; portable replay uses the archived complete decoded stacks.

Invalid/retired prefixes remain committed (graph timing/epoch/capacity,
client_dispatch_initial/,official retries,finite-reference flags). They are
excluded only as documented in original plans/results.
Model weights, credentials, environments and compiled shared libraries are not
included and not needed offline. Pinned public model download is below.
Absolute strings in immutable historical logs are provenance,not paths to run.

## Live reruns

Hardware:RTX5090(sm_120),Core Ultra9 285K/24logical CPUs,one separate Linux
x86-64 TCP client (originally g10). See [HOST](bench/results/HOST.md).
Software:Linux7.3-rc3 with BTF/cgroupv2/bpffs/tested io_uring BPF/UAPI;
driver610.57.04;CUDA toolkit13.0/nvcc13.0.88;clang,C++,make,libelf/zlib.
Inference:Python3.12,torch2.13.0+cu129,vLLM0.31.0 and
[requirements.txt](requirements.txt). Remote LLM client needs uvloop/aiohttp;
remote graph client needs Python3.12 plus the native binary.

Actual ordinary builds use system libbpf1.3 headers with libbpf1.6.3
link/runtime library. Adding1.6.3 headers via C_INCLUDE_PATH/CPLUS_INCLUDE_PATH
conflicts with bpf_stream_vprintk in generated vmlinux.h. The failed clean
build is retained. Point LIBRARY_PATH/LD_LIBRARY_PATH to installed1.6.3 libraries
when necessary;leave headers on the qualified system path. This is not a
fresh-OS/arbitrary-headers claim. No kernel/driver/global-CPU mutation is needed.

### Short ordinary graph smoke

~~~sh
../artifact-env/bin/python -m pip install 'torch==2.13.0+cu129' \
  --extra-index-url https://download.pytorch.org/whl/cu129
PATH="../artifact-env/bin:$CUDA_HOME/bin:$PATH" \
  ../artifact-env/bin/python bench/smoke_paper.py --output ../gpu-smoke
~~~

CUDA_HOME is your installed toolkit. This builds ordinary targets and tests
both transports with cuBLAS/cuFFT:4cells/64exact independent constant-input
oracles, persistent reuse, normal stop/detach. About15–30s after dependencies;
allow2min. It is build/correctness qualification, not statistical evidence.
It refuses existing BPF owners and cleans only its own attachments.
Actual new-environment execution is recorded, including the initial failed
header selection. GPU primitives and host stack are prerequisites,not emulated.

### Graph screen/confirmation:Table2/Figure2

RUN is a new private output directory and BUILD its build directory.
Use the matching Python environment and library paths.

~~~sh
make BUILD="$BUILD" "$BUILD/server.bpf.o" "$BUILD/bpf_gpu_server_load" \
  graph-server-kernel-network graph-server-user-network-wait graph-network-client
"$BUILD/bpf_gpu_server_load" stream-uring-attach
BGS_BIND="$SERVER_IP" python3 examples/torch_graph_server.py \
  "$BUILD/graph_server_kernel_network.so" fft --repeat 1 --seconds 600
~~~

For native substitute graph_server_user_network_wait.so without attach.
After ready, record the PID and run bench/resource_observer.py --pid "$PID"
--output "$RUN/cell.resources.jsonl" (dependency:nvidia-ml-py),including the
separate idle bracket from the archived cell lifecycle.
Copy graph_network_client,bench/graph_network_client.py and its .cpp into
one new directory on the remote client:

~~~sh
python3 graph_network_client.py --host "$SERVER_IP" --operation fft \
  --rate 1000 --seconds 6 --seed 320261017 --output "$CLIENT_RUN/cell.client.jsonl"
~~~

Defaults:8workers×16connections;--native accepts an explicit binary path.
After drain SIGINT your server/SIGTERM your observer,wait normal exits,then
detach/remove only owned pins using the [runbook](docs/graph-backends.md).

Full schedule:FFT/GEMM,repeats1/16/256,allCPUs;repeat1 also own-service
affinity1/2. Sweep1000/4000/16000rps,6s/cell,predeclared saturation stop.
Preserve exact seeds,order,brackets and criteria from the archived
[screen](bench/results/graph_network_20261017/run_graph_network_screen.ps1),
[confirmation](bench/results/graph_network_20261017/run_graph_network_confirmation.ps1)
and [cell](bench/results/graph_network_20261017/run_graph_network_cell.ps1)
controllers. Their cluster access is historical;replace only that access with
your own server/client invocation. They are not portable infrastructure setup.

Confirm the already frozen FFT/all,GEMM/all,GEMM/one-CPU points at repeat1/
1000rps with5new-seed counterbalanced20s pairs each. Do not reselect.
60screen cells/2.52M requests plus30confirmation/600k;offered windows16min,
allow30–60min with startup/observer/drain. Expected:zero observed failures on
qualified stack,near1000rps confirmation,CPU ratios near1 with broad intervals.
No guaranteed P99 sign or production tolerance. Both transports first
saturated at16000 after4000. The4-cell smoke is the short correctness version.

The LLM smoke also accepts the pinned SmolLM2-135M snapshot using the same
--model argument and four conditions; it selects that checkpoint’s archived
prompt/output references rather than applying Qwen token IDs.

### LLM final matrices:Table1/capacity

~~~sh
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir ../models/qwen
make BUILD="$BUILD16" CPPFLAGS=-DBGS_LLM_SLOTS=16 \
  "$BUILD16/server.bpf.o" "$BUILD16/bpf_gpu_server_load" \
  vllm-mixed-kernel-network vllm-mixed-user-network-wait
make BUILD="$BUILD32" CPPFLAGS=-DBGS_LLM_SLOTS=32 \
  "$BUILD32/server.bpf.o" "$BUILD32/bpf_gpu_server_load" vllm-mixed-user-network-wait
BGS_BIND="$SERVER_IP" PYTHONPATH=executor \
  "$BUILD16/vllm_mixed_user_network_wait" ../models/qwen 600
~~~

Substitute BUILD32 for native32. Kernel16:attach with its matching loader,
then launch vllm_mixed_kernel_network. See
[ordinary start/cleanup](docs/getting-started.md). Current main integrates
the formerly private candidate;immutable captured sources define historical
Table1 top. External optimized reference:

~~~sh
vllm serve ../models/qwen --tokens-only --dtype half --host "$SERVER_IP" --port 39404
~~~

Use an ephemeral private-test API key;pass its file via --key-file,never commit
it. On the remote client export the first original raw client's config dictionary
into a new config.json,copy that experiment's remote_offsets.json and its
official_remote_client.py (uvloop/aiohttp required):

~~~sh
python3 official_remote_client.py --host "$SERVER_IP" \
  --config "$CLIENT_RUN/config.json" --offsets remote_offsets.json \
  --output "$CLIENT_RUN/cell.client.jsonl" --pair 0 --rate 192 --mode native
~~~

Use --mode kernel/official as appropriate and --warmup separately first.
The original transport client --preflight uses32 attempts;the later capacity
client uses128. Both request64 tokens;neither estimates
P99/capacity. A standalone local smoke builds both capacities and exercises
native16,kernel16,native32 and official with the same archived client:

~~~sh
python3 -m pip install -r requirements.txt
python3 bench/smoke_paper_llm.py --model ../models/qwen --output ../llm-smoke
~~~

Run this in a separate new inference environment,with CUDA/library paths set
as above. Allow2–10min for fresh graph capture and official compilation.
It checks128 protocol-valid requests/8192 output tokens plus32 warmups;
finite-reference flags are counted rather than hidden. It is local correctness/
workload smoke,not the remote performance experiment or universal exact fidelity.

Full protocol:5blocks×3conditions×60s at192rps plus drain,same4prompts/gen64,
original normalized arrivals/counterbalance. Controllers:
[transport](bench/results/official_sustained_20261016/run_official_remote_matrix.ps1),
[capacity](bench/results/resident_queue_relief_20261017/run_queue_relief_matrix.ps1).
Keep each experiment's observer/client/frame/timeout rules.
Allow1–2h per matrix including first capture(model download extra),15min
offered windows. Historical resident16≈11.6k token/s/multi-second tails;
native32/official≈12.25k/hundreds-of-ms tails,near12288offered ceiling.
These observations are not guaranteed tolerances on a new machine.

### Historical fixes and perf:Table3/Figure3

Current main contains the fixes. Export source-only historical snapshots into
private directories to reconstruct ablations;do not change the working main.
Captured source/build arguments and frozen directories take precedence over
the contemporaneous commit ID when candidates were private before integration.

| Group | Original acquisition entrypoint under R | Protocol,time,expected scale |
|---|---|---|
| Mixed0065 | mixed_batching_20261015/load.py --model "$MODEL" --output "$RUN/full_qwen.jsonl";archived original/mixed constructors,16-bank BGS_TRACE builds;--preflight uses32attempts/cell | 4conditions×2rates96/192×5blocks×256;Qwen and Smol originally;1–2h;Qwen96 TPOT≈1.43→1.23ms,native also improves |
| NODELAY0070 | tcp_nodelay_20261016/tcp_nodelay.py --output "$RUN/full.jsonl";before/after nodelay_build.py;--preflight32attempts/cell with diagnostic packets | 5blocks×4conditions×2regimes×256=10240requests;30–60min;closed16gap≈40→2–3ms,CPU rises |
| Reclamation0072 | connection_reclamation_20261016/reclamation_server.py,verified_reclaim_build.py and the [plan](bench/results/connection_reclamation_20261016/plan.md) | global/local/native15×60s/192rps;1–2h;historical4157timeouts→0,count varies;global guard is a defined ablation |
| Active wait0074 | active_admission_wait_20261016/run_active_wait_matrix.ps1 and active_wait_server.py;uring_loop.before.bpf.c vs uring_loop.candidate.bpf.c,same16-slot GPU SASS | 15×60s/172800attempts;1–2h;CPU≈1.70→.63ms;earlier18.7% native gap did not reproduce in final path |
| Perf0073 | command below;repaired pre-active-wait16-slot sources from kernel_cpu_profile_20261016/source-context.md and server receipts | one ordered native/kernel pair,2×60s/192rps;allow15min startup/decode;10086/2455samples and71/85% are historical,not exact targets |

~~~sh
perf record -e cpu-clock:k -F 499 --call-graph fp --tid "$CREATOR_TID" \
  --no-inherit --no-buildid --no-buildid-cache -o "$RUN/perf.data"
~~~

After full drain SIGINT your recorder and retain its actual SIGINT exit.
Use original decode-field arguments from retained receipts,then
profile_summary.py;decode/compress after both serving cells.
Historical drivers encode old runtime/model/build/cluster paths. Stage their
helpers,set r to your source root,t to your private runtime root,and relocate
environment/model values to your installed dependencies/pinned model.
Never execute into immutable archived result directories.
Plans,expanded commands and captured sources are the historical recipe.
Fully automatic portable historical reacquisition is not claimed;the single
command guarantee covers offline replay,and the executed new-environment
smoke covers the ordinary current graph path.

## Verification and daily upkeep

The [validation record](bench/results/paper_artifact_20261017/README.md)
contains actual clean source/environment commands,durations,exits,
download/input-availability checks and cleanup. Implementation,experiments,
writing and independent-code audit all stay in this thread,without delegation.

Daily reflection:the bounded performance search is closed. The paper's current
blocker is reproducibility and claim precision. The most useful next step is
raw-data arithmetic,an honest discrepancy list and a clean-run receipt,not
another search for favorable performance. README now links this artifact;
historical data and foreign WIP remain.
