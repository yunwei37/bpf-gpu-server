#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: external HTTP baseline client (vLLM OpenAI-compatible server).
#
# The fair "what a normal serving stack does" path: a streamed chat completion
# over TCP/HTTP. TTFT = request write -> first content token; TPOT = mean
# inter-token gap of the content tokens. Only SSE data frames with a non-empty
# content delta count as tokens, matching the control-plane clients.

import argparse, json, socket, time

HOST = '127.0.0.1'
PORT = 8000
PROMPT = 'The capital of France is'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

def measure(host, port, prompt, n_gen, timeout=60):
    body = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': n_gen, 'temperature': 0.0, 'stream': True,
    }).encode()
    req = (f'POST /v1/chat/completions HTTP/1.1\r\n'
           f'Host: {host}:{port}\r\n'
           f'Content-Type: application/json\r\n'
           f'Content-Length: {len(body)}\r\n'
           f'Connection: close\r\n\r\n').encode() + body
    t0 = time.perf_counter()
    s = socket.create_connection((host, port), timeout=timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.sendall(req)
    buf = b''
    stamps = []
    while len(stamps) < n_gen:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            frame, buf = buf.split(b'\n\n', 1)
            for line in frame.split(b'\n'):
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                ch = obj.get('choices')
                if not ch:
                    continue
                delta = ch[0].get('delta', {})
                content = delta.get('content')
                if content:
                    stamps.append(time.perf_counter())
                    if len(stamps) >= n_gen:
                        break
    s.close()
    if not stamps:
        return None
    ttft = (stamps[0] - t0) * 1e6
    tpot = ((stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1e6
            if len(stamps) > 1 else 0.0)
    return ttft, tpot, len(stamps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=HOST)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--prompt', default=PROMPT)
    ap.add_argument('--gen', type=int, default=32)
    ap.add_argument('--rounds', type=int, default=6)
    a = ap.parse_args()
    global MODEL
    MODEL = a.model
    measure(a.host, a.port, a.prompt, 4)  # warm
    tt, tp = [], []
    text = ''
    for _ in range(a.rounds):
        r = measure(a.host, a.port, a.prompt, a.gen)
        if r is None:
            continue
        tt.append(r[0]); tp.append(r[1])
    if not tt:
        print('http baseline: no samples')
        return
    tt.sort(); tp.sort(); n = len(tt)
    print(f'vllm-http TTFT us: p50={tt[n//2]:.0f} min={tt[0]:.0f} '
          f'max={tt[-1]:.0f} | TPOT us: p50={tp[n//2]:.1f} '
          f'min={tp[0]:.1f} max={tp[-1]:.1f}')


if __name__ == '__main__':
    main()
