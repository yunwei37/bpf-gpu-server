# Pillow plotting dependency repair, October 23

GitHub's push response exposed nine open dependency alerts: one critical, six
high and two medium. All identify Pillow12.3.0 in the historical
[plot freeze](../sustained_remote_resources_20261016/plot_requirements.txt).
[advisories.json](advisories.json) records each advisory, affected range and
first patched version. Pillow13.0.0 falls outside all nine reported ranges.
This is a manifest finding, not evidence of exploitation or a live server fault.

The active [artifact requirements](../../artifact-requirements.txt) now pin
Pillow13.0.0. The archived freeze remains byte-identical. Do not install that
historical freeze as the current replay environment. The nine archive alerts
remain open; none was dismissed or hidden. This does not establish that every
package or server path is vulnerability-free.

## Reproduce

From the source root, using Python3.12 with venv/ensurepip support:

~~~sh
python3 -m venv ../artifact-env
../artifact-env/bin/python -m pip install -r bench/artifact-requirements.txt
../artifact-env/bin/python bench/reproduce_paper.py --output ../artifact-output
~~~

Use new environment/output directories. This run used a fresh temporary
Python3.12.3 environment in the original Workspace. Its image lacks ensurepip
and an existing pip, so setup used venv --without-pip and a PyPI pip wheel via
PYTHONPATH to install into that environment only. No global package or serving
runtime was changed. [environment.json](environment.json) retains installed
package versions; there was no Torch/vLLM, and CUDA_VISIBLE_DEVICES was empty.
The replay itself rejects socket operations.

## Result

[comparison.json](comparison.json) and [comparison.md](comparison.md):
136PASS, zero mismatches, zero changed-paper checks,214.839s,7182source inputs.
[figure_data.json](figure_data.json) is byte-identical to the
[post-rename result](../rename_20261017/offline/figure_data.json).
The regenerated [CPU plot](figure2_cpu.png) and [perf plot](figure3_perf.png)
were visually inspected; PDF versions are included. [replay.log](replay.log)
uses a relative output path in place of the ephemeral absolute directory.

These are new numerical replay/plotting compatibility results. No GPU or
performance experiment was repeated. Original measurement files and paper
text/PDF were unchanged. The temporary environment, bootstrap wheel, caches
and output directory were removed after copying these receipts.
