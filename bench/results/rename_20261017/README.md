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

Two-model LLM and full offline-artifact replay follow this source checkpoint;
their completion is not claimed here. All implementation/review is in this
thread, with no delegated agent or new performance experiment.
