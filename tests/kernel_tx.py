#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live kernel-TX protocol regression: no model inference, synthetic token IDs.
Run from the root after make probes tools and make -C module CC=gcc-15.
Owns only this namespace's bpf-gpu-server tc hook and test module for the run.
"""
import ctypes
import errno
import socket
import struct
import subprocess
import time
from llm_executor import (open_page, Page, LLM_HEAD_OFF, BGS_LLM_SLOTS,
                          LLM_RING_OFF, LLM_SLOT_SIZE, BGS_PENDING, BGS_FREE)


def wait_for(fn):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.001)
    raise AssertionError("protocol progress timeout")


def main():
    subprocess.run(["./build/bpf_gpu_server_load", "detach", "lo"], check=True)
    subprocess.run(["./build/bpf_gpu_server_load", "attach", "lo"], check=True)
    loaded = False
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 39403))
    listener.listen()
    listener.settimeout(5)
    mm, base = open_page()
    page = Page(mm, base)
    try:
        # Reject a genuine but incompatible BPF map FD before array casting.
        lib = ctypes.CDLL(None, use_errno=True)
        libbpf = ctypes.CDLL("libbpf.so.1", use_errno=True)
        wrong = libbpf.bpf_obj_get(b"/sys/fs/bpf/bgs_stats")
        assert wrong >= 0
        with open("module/bpf_gpu_server_tx.ko", "rb") as module:
            ret = lib.syscall(313, module.fileno(), f"map_fd={wrong}".encode(), 0)
            assert ret == -1 and ctypes.get_errno() == errno.EINVAL
        import os
        os.close(wrong)
        print("PASS wrong map rejected", flush=True)
        subprocess.run(["./build/bpf_gpu_server_load", "tx-load"], check=True)
        loaded = True
        for trial in range(26):
            head = page.u32(LLM_HEAD_OFF)
            idx = head % BGS_LLM_SLOTS
            assert page.slot(idx)[0] == BGS_FREE
            client = socket.create_connection(("127.0.0.1", 39403), timeout=5)
            server, _ = listener.accept()
            n = 1 if trial == 0 else 64
            marker = time.monotonic_ns()
            client.sendall(struct.pack("<IIIIQI", 0x514c4d51, 1, n, 0, marker, 42))
            wait_for(lambda: page.u32(LLM_HEAD_OFF) == head + 1)
            assert page.slot(idx)[7] == marker
            disconnected = trial in (7, 17)
            if disconnected:
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                client.close()
            expected = b"".join(struct.pack("<I", trial * 1000 + k) for k in range(n))
            for k in range(n):
                page.set_tok_out(idx, k, trial * 1000 + k)
                page.set_produced(idx, k + 1)
            time.sleep(0.08) # exceeds the former 500 * 60 us reclaim interval
            assert page.slot(idx)[0] == BGS_PENDING, "reclaimed while executor owned slot"
            data = b""
            if not disconnected:
                # The existing TX contract retains the final token until DONE.
                while len(data) < len(expected) - 4:
                    chunk = client.recv(1)
                    assert chunk
                    data += chunk
                assert data == expected[:-4]
                assert page.slot(idx)[0] == BGS_PENDING
            page.finish(idx)
            if not disconnected:
                while len(data) < len(expected):
                    # Keep the original single-byte exact stream check.
                    chunk = client.recv(1)
                    assert chunk
                    data += chunk
                assert data == expected
            wait_for(lambda: page.slot(idx)[0] == BGS_FREE)
            if not disconnected:
                client.close()
            server.close()
        print("PASS 24 exact TCP replies, 2 resets, 26 DONE/recycle cycles, PENDING preserved", flush=True)
    finally:
        try:
            if loaded:
                ctypes.set_errno(0)
                ret = lib.syscall(176, ctypes.c_char_p(b"bpf_gpu_server_tx"), 0)
                if ret != 0:
                    raise OSError(ctypes.get_errno(), "delete_module")
        finally:
            listener.close()
            mm.close()
            subprocess.run(["./build/bpf_gpu_server_load", "detach", "lo"], check=True)
            import pathlib
            for name in ("bgs_stats", "bgs_db", "bgs_ctl"):
                pathlib.Path("/sys/fs/bpf", name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
