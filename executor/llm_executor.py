#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: resident LLM executor.
#
# Keeps Qwen loaded on the GPU and spins on the pinned BPF page's LLM ring. For
# each PENDING slot it tokenizes are already-decoded ids from the page, runs
# prefill + decode, and streams each generated token id into slot.tok_out,
# bumping `produced` (release) after every token so a client watching the page
# measures TTFT (first token) and TPOT (inter-token) over the kernel->page path.
#
# The page is opened with bpf_obj_get + mmap (the same pinned object the C
# executor uses), so this is the same queue, not a side channel.
import argparse, ctypes, mmap, os, socket, struct, sys, threading, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSFS = "/sys/fs/bpf/bgs_ctl"

BGS_LLM_SLOTS = 8
BGS_LLM_MAX_TOK = 64
BGS_LLM_TCP_PORT = 39403

# Must match struct bgs_llm_slot in bpf/include/bpf_gpu_server_queue.h, in order.
#   u32 state, n_gen, n_prompt, produced, addr_be; u16 port_be, pad;
#   u64 client_ns, ingress_ns, gpu_done_ns; u32 tok_in[64]; u32 tok_out[64];
LLM_SLOT_FMT = "<IIIIIHHQQQ64I64I"
LLM_SLOT_SIZE = struct.calcsize(LLM_SLOT_FMT)
assert LLM_SLOT_SIZE == 4 * 5 + 4 + 24 + 256 + 256, LLM_SLOT_SIZE

# struct bgs_page: 9 u32 + u64 stop_ns, then slots/peers/done, then llm_head.
# Offsets of the LLM ring are computed from the C-layout sizes.
CTL_SLOT_SIZE = 4 + 4 + 8 + 8 + 16 * 4           # state,id,client,ingress,x[]
DONE_SLOT_SIZE = 4 + 4 + 8 + 8 + 8 + 8 + 16 * 4  # id,state,client,ingress,start,done,y[]
PEER_SIZE = 8
BGS_SLOTS = 64
PAGE_HDR = 9 * 4 + 4 + 8                         # 9 u32s, 4 pad, u64 stop_ns
LLM_HEAD_OFF = PAGE_HDR + BGS_SLOTS * (CTL_SLOT_SIZE + PEER_SIZE + DONE_SLOT_SIZE)
LLM_RING_OFF = LLM_HEAD_OFF + 8                  # llm_head(4) + llm_reserved(4)
PAGE_BYTES = LLM_RING_OFF + BGS_LLM_SLOTS * LLM_SLOT_SIZE
MAP_BYTES = (PAGE_BYTES + 4095) // 4096 * 4096

BGS_PENDING = 1
BGS_FREE = 0
BGS_DONE = 2

# The reference CPU bridge must preserve concurrent producer reservations when
# recycling a slot, just like kernel TX. GCC's native atomic operation also
# releases earlier mmap stores; a Python read/modify/write could lose a head.
_atomic = ctypes.CDLL("libatomic.so.1")
_release_reservation = getattr(_atomic, "__atomic_fetch_and_8")
_release_reservation.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
_release_reservation.restype = ctypes.c_uint64


def open_page():
    libbpf = ctypes.CDLL("libbpf.so.1")
    fd = libbpf.bpf_obj_get(SYSFS.encode())
    if fd < 0:
        sys.exit(f"bpf_obj_get({SYSFS}): errno {ctypes.get_errno()} "
                 "(run bpf_gpu_server_load attach)")
    buf = mmap.mmap(fd, MAP_BYTES, mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE)
    return buf, ctypes.addressof(ctypes.c_char.from_buffer(buf))


class Page:
    def __init__(self, mm, base):
        self.mm = mm
        self.base = base

    def u32(self, off):
        return struct.unpack_from("<I", self.mm, off)[0]

    def slot(self, idx):
        off = LLM_RING_OFF + idx * LLM_SLOT_SIZE
        v = struct.unpack_from(LLM_SLOT_FMT, self.mm, off)
        (state, n_gen, n_prompt, produced, addr_be, port_be, pad,
         client_ns, ingress_ns, gpu_done_ns) = v[:10]
        return (state, n_gen, n_prompt, produced, addr_be, port_be, pad,
                client_ns, ingress_ns, gpu_done_ns,
                v[10:74], v[74:138])

    def set_produced(self, idx, val):
        off = LLM_RING_OFF + idx * LLM_SLOT_SIZE + 12   # produced
        struct.pack_into("<I", self.mm, off, val)

    def set_tok_out(self, idx, k, val):
        off = LLM_RING_OFF + idx * LLM_SLOT_SIZE + 4 * 12 + 256 + k * 4
        struct.pack_into("<I", self.mm, off, val)

    def finish(self, idx):
        off = LLM_RING_OFF + idx * LLM_SLOT_SIZE
        struct.pack_into("<Q", self.mm, off + 40, time.monotonic_ns())
        struct.pack_into("<I", self.mm, off, BGS_DONE)

    def recycle(self, idx):
        off = LLM_RING_OFF + idx * LLM_SLOT_SIZE        # state
        struct.pack_into("<I", self.mm, off, BGS_FREE)
        _release_reservation(self.base + LLM_HEAD_OFF,
                             ctypes.c_uint64(~(1 << (32 + idx))).value, 3)


def tcp_accept_loop(lsock, conns, lock, stop):
    """Accept client connections and keep them in a {peer_key: sock} map so the
    executor can stream tokens back on the socket the request arrived on."""
    lsock.settimeout(0.2)
    while not stop[0]:
        try:
            c, peer = lsock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        key = (peer[0], peer[1])
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with lock:
            conns[key] = c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--eager", action="store_true", help="disable CUDA graphs / compile")
    ap.add_argument("--tcp", action="store_true",
                    help="stream tokens back over the request's TCP socket")
    ap.add_argument("--tcp-bind", default="0.0.0.0")
    ap.add_argument("--no-send", action="store_true",
                    help="do not send() from userspace; the bpf_gpu_server_tx kernel "
                         "module transmits the reply (module/kmod)")
    a = ap.parse_args()

    if a.no_send and not a.tcp:
        ap.error("--no-send requires --tcp")

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    dt = torch.float16 if a.dtype == "float16" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()

    mm, base = open_page()
    page = Page(mm, base)
    print(f"executor: model loaded, page {base:#x}, llm ring off {LLM_RING_OFF}, "
          f"slot {LLM_SLOT_SIZE}B", flush=True)

    conns = {}
    conns_lock = threading.Lock()
    stop = [False]
    if a.tcp:
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((a.tcp_bind, BGS_LLM_TCP_PORT))
        lsock.listen(64)
        threading.Thread(target=tcp_accept_loop,
                         args=(lsock, conns, conns_lock, stop),
                         daemon=True).start()
        print(f"executor: TCP listener on {a.tcp_bind}:{BGS_LLM_TCP_PORT}",
              flush=True)

    def generate(prompt_ids, n_gen):
        """Yield generated ids one at a time."""
        ids = torch.tensor([prompt_ids], device=dev)
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[:, -1].argmax(-1))
            yield nxt
            for _ in range(n_gen - 1):
                out = model(torch.tensor([[nxt]], device=dev),
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[:, -1].argmax(-1))
                yield nxt

    t_end = time.monotonic() + a.seconds
    seen = 0
    served = 0
    while time.monotonic() < t_end:
        head = page.u32(LLM_HEAD_OFF)
        if head == seen:
            continue
        if head - seen > BGS_LLM_SLOTS:
            seen = head - BGS_LLM_SLOTS
        idx = seen % BGS_LLM_SLOTS
        (state, n_gen, n_prompt, produced, addr_be, port_be, pad,
         client_ns, ingress_ns, gpu_done_ns, tok_in, tok_out) = page.slot(idx)
        if state != BGS_PENDING:
            # Head reserves this sequence before its producer publishes PENDING.
            continue
        prompt = list(tok_in[:n_prompt])
        # For a TCP request, find the accepted connection this segment came
        # from so tokens can be streamed back on the same socket.
        conn = None
        if pad == 1 and not a.no_send:
            key = (socket.inet_ntoa(struct.pack("<I", addr_be)),
                   socket.ntohs(port_be))
            # The accept thread may not have caught up with the segment yet;
            # give it a moment before giving up on the reply socket.
            for _ in range(50):
                with conns_lock:
                    conn = conns.get(key)
                if conn is not None:
                    break
                time.sleep(0.002)
        t0 = time.perf_counter()
        k = 0
        for tid in generate(prompt, n_gen):
            page.set_tok_out(idx, k, tid)
            k += 1
            page.set_produced(idx, k)   # release: client/module sees the token
            if conn is not None and not a.no_send:
                try:
                    conn.sendall(struct.pack("<I", tid))
                except OSError:
                    conn = None
            if k >= n_gen:
                break
        if a.no_send and pad == 1:
            page.finish(idx)    # release ownership to kernel TX
        else:
            page.recycle(idx)
        t1 = time.perf_counter()
        seen += 1
        served += 1
        if served % 20 == 0:
            print(f"executor: served={served} last gen={k} tok "
                  f"{(t1-t0)*1e3:.1f} ms", flush=True)

    stop[0] = True
    print(f"executor: exit served={served}", flush=True)


if __name__ == "__main__":
    main()
