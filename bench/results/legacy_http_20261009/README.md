# Retired October 9 HTTP baseline tools

Verbatim sources moved here on November3 from bench/run_http.sh and
tools/llm_http_client.py, parent revision13fb37c8. No source bytes were changed.
These are historical evidence for [finding0011](../../../docs/findings/0011-http-baseline.md),
not supported launch commands. The old driver used global pkill, no API-key
configuration and a shared temporary log; its client counted SSE chunks rather
than verified output tokens. Later findings0020/0021 already recorded these
limitations. Original raw measurements and finding text remain unchanged.

Use the bounded official LLM smoke/matrix instructions in the
[current artifact](../../../ARTIFACT.md#llm-final-matricestable1capacity).
No old or current server was launched or signalled during this retirement.
