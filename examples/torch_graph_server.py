#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture upstream cuBLAS/cuFFT graphs and hand opaque buffers to bpf-gpu-server.

Use the existing TCP/HTTP frame, with input and output lengths both 64.
Words represent IEEE float32 bit patterns, without numeric conversion.
Matmul: row-major 8x8 FP32 input times fixed W[i,j]=(8*i+j)%7-3.
FFT: 32 complex64 values, interleaved real/imaginary FP32 words.
"""
import argparse
import ctypes
from pathlib import Path
import torch

def capture_graphs(operation,slots,repeat=1):
    graphs=[];inputs=[];outputs=[];weights=[]
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream),torch.inference_mode():
        for _ in range(slots):
            if operation=="matmul":
                x=torch.zeros((8,8),device="cuda",dtype=torch.float32)
                weight=(torch.arange(64,device="cuda",dtype=torch.float32)%7-3).reshape(8,8)
                y=torch.empty_like(x)
                def run(): torch.mm(x,weight,out=y)
                weights.append(weight)
            else:
                x=torch.zeros(32,device="cuda",dtype=torch.complex64)
                y=torch.empty_like(x)
                def run(): torch.fft.fft(x,out=y)
            for _ in range(3): run()
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):
                for _ in range(repeat): run()
            graphs.append(graph);inputs.append(x);outputs.append(y)
    stream.synchronize()
    return graphs,inputs,outputs,weights

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library",type=Path)
    parser.add_argument("operation",choices=["matmul","fft"])
    parser.add_argument("--seconds",type=int,default=3600)
    parser.add_argument("--repeat",type=int,default=1,help="library operations per captured graph; input/output unchanged")
    args=parser.parse_args()
    if args.seconds<=0: parser.error("--seconds must be positive")
    if args.repeat<=0: parser.error("--repeat must be positive")
    lib=ctypes.CDLL(str(args.library.resolve()))
    for name in ["bgs_graph_abi_version","bgs_graph_max_slots","bgs_graph_max_words"]:
        getattr(lib,name).restype=ctypes.c_uint
    if lib.bgs_graph_abi_version()!=1: raise RuntimeError("unsupported graph ABI")
    if lib.bgs_graph_max_words()<64: raise RuntimeError("example needs 64 words")
    slots=lib.bgs_graph_max_slots()
    graphs,inputs,outputs,weights=capture_graphs(args.operation,slots,args.repeat)
    Array=ctypes.c_uint64*slots
    graph_ptrs=Array(*(g.raw_cuda_graph() for g in graphs))
    input_ptrs=Array(*(x.data_ptr() for x in inputs))
    output_ptrs=Array(*(y.data_ptr() for y in outputs))
    lib.bgs_graph_serve.argtypes=[ctypes.POINTER(ctypes.c_uint64)]*3+[ctypes.c_uint]*3
    lib.bgs_graph_serve.restype=ctypes.c_int
    result=lib.bgs_graph_serve(graph_ptrs,input_ptrs,output_ptrs,64,64,args.seconds)
    if result: raise SystemExit(result)

if __name__=="__main__":
    main()
