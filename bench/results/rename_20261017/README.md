# bpf-gpu-server namespace migration

Reader branding was committed asaf91e8eb. The source migration changes loader,
shared header, module, macro/environment, BPF pin and graph-export names; the
[current mapping](../../../README.md) lists the public changes. Frozen raw data
and historical command examples retain their names. Paper files are owned
separately and unchanged by this thread.

Completed source checkpoint: ordinary make all plus both graph and mixed-vLLM
transport targets pass. The [graph smoke](graph-smoke/smoke.json) passes four
cuBLAS/cuFFT/native/kernel cells and64 exact replies. Build logs are captured.

The legacy module also compiles with the already installed matching kernel
headers and gcc15; it is not loaded. The first default module attempt records
the absent /lib/modules build link. The next copied-source attempt omitted its
relative shared headers; after copying those same headers, the qualified
[build](module-build-qualified.log) exits0. Kernel/compiler/pahole warnings
remain visible. No host package, symlink or configuration changes are made.

The renamed-source [full offline replay](offline/comparison.md) now completes
136PASS/0MISMATCH/0PAPER_CHANGED, exit0 in208.994s in a new plotting environment
without GPU/inference packages. All7182input paths and both figures’ numerical
data are byte-identical to the preceding paper-owner-update replay. Both data
figures are regenerated. The small veth example’s remaining short bfns/bfv
labels are renamed to bgsns/bgsv and its shell syntax passes; it is not run.
[Qwen smoke](qwen-smoke/smoke.json) now passes all four conditions:
native16/kernel16/native32/official,128requests/8192tokens and32excluded gen1
warmups. All servers exit0 and this run has0finite-reference disagreements.
Raw clients/workers and sanitized server logs are retained. The initial omitted-
venv-PATH failure happens before readiness or requests; its logs are retained in
qwen-initial-environment-failure/. Documented environment activation fixes the
controller invocation; serving source/criteria are unchanged. The ordinary
Make graph-network-client target also builds; direct-compiler setup failures
are retained. Smol smoke remains running at this checkpoint. All implementation/review is in this
thread, with no delegated agent or new performance experiment.
