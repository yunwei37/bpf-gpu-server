#!/usr/bin/env python3
"""CPU/resource observer with GPU queries only at window boundaries."""
import argparse,ctypes,json,os,signal,time
from pathlib import Path
import pynvml as nv
libc=ctypes.CDLL(None);libc.clock_getcpuclockid.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
done=False
def stop(*_):
 global done;done=True
def read(p):
 try:return Path(p).read_text()
 except OSError as e:return dict(unavailable=repr(e))
def group(pid):
 result={}
 for p in Path("/proc").glob("[0-9]*/stat"):
  try:
   text=p.read_text();v=text.rsplit(") ",1)[1].split()
   if int(v[2])!=pid:continue
   clock=ctypes.c_int()
   if libc.clock_getcpuclockid(int(p.parent.name),ctypes.byref(clock)):continue
   result[p.parent.name]=dict(stat=text,cpu_ns=time.clock_gettime_ns(clock.value),status=read(p.parent/"status"),cgroup=read(p.parent/"cgroup"))
  except (OSError,ValueError):continue
 return result
def threads(pid):
 result={}
 for p in Path("/proc/%d/task"%pid).glob("[0-9]*"):
  try:result[p.name]={f:(p/f).read_text() for f in ["stat","status","schedstat","comm"]}
  except OSError:pass
 return result
def map_metadata():
 result={}
 try:
  bpf=ctypes.CDLL("libbpf.so.1",use_errno=True)
  for p in Path("/sys/fs/bpf").glob("bgs_*"):
   fd=bpf.bpf_obj_get(str(p).encode())
   if fd<0:result[p.name]=dict(unavailable=ctypes.get_errno());continue
   try:
    info=read("/proc/self/fdinfo/%d"%fd)
    result[p.name]=dict(fdinfo=info)
    if isinstance(info,str) and "map_type:" in info:
     data=ctypes.create_string_buffer(256);length=ctypes.c_uint(256)
     status=bpf.bpf_obj_get_info_by_fd(fd,data,ctypes.byref(length))
     result[p.name].update(info_exit=status,info_length=length.value,info_hex=data.raw[:length.value].hex() if status==0 else None)
   finally:os.close(fd)
 except OSError as e:result["unavailable"]=repr(e)
 return result
def main(args):
 nv.nvmlInit();h=nv.nvmlDeviceGetHandleByIndex(0)
 cg=Path("/sys/fs/cgroup")/Path("/proc/%d/cgroup"%args.pid).read_text().strip().split("::",1)[1].lstrip("/")
 assert cg.is_dir(),str(cg)
 raw=Path(args.output).open("x",buffering=1);count=0
 def emit(event,**kw):raw.write(json.dumps(dict(event=event,mono_ns=time.monotonic_ns(),**kw))+"\n")
 def call(fn,*a):
  try:return fn(*a)
  except nv.NVMLError as e:return dict(unavailable=type(e).__name__)
 def sample(detail,gpu=False):
  before=time.monotonic_ns();device={}
  if gpu:
   device["energy_mJ"]=call(nv.nvmlDeviceGetTotalEnergyConsumption,h)
   device["power_mW"]=call(nv.nvmlDeviceGetPowerUsage,h)
   for name,kind in [("sm_clock_MHz",nv.NVML_CLOCK_SM),("memory_clock_MHz",nv.NVML_CLOCK_MEM)]:
    device[name]=call(nv.nvmlDeviceGetClockInfo,h,kind)
   util=call(nv.nvmlDeviceGetUtilizationRates,h)
   device["utilization"]={k:getattr(util,k) for k in ["gpu","memory"]} if not isinstance(util,dict) else util
   memory=call(nv.nvmlDeviceGetMemoryInfo,h)
   device["memory_bytes"]={k:getattr(memory,k) for k in ["total","used","free"]} if not isinstance(memory,dict) else memory
   apps=call(nv.nvmlDeviceGetComputeRunningProcesses,h)
   device["compute_apps"]=[dict(pid=x.pid,usedGpuMemory=x.usedGpuMemory) for x in apps] if not isinstance(apps,dict) else apps
  out=dict(sample_begin_ns=before,sample_end_ns=time.monotonic_ns(),device=device,process_group=group(args.pid),
    host={f:read("/proc/"+f) for f in ["stat","softirqs","interrupts","loadavg","pressure/cpu","pressure/io","pressure/memory"]},
    cgroup={f:read(cg/f) for f in ["cpu.stat","cpu.pressure","cpu.max","cpu.weight","memory.current","memory.events","io.stat"]},
    observer_cpu_ns=time.process_time_ns(),observer_stat=read("/proc/self/stat"))
  if detail:
   out["service_threads"]=threads(args.pid)
   # Host kthreads visible in this namespace are incomplete; absence is unknown.
   out["visible_kernel_workers"]={}
   for p in Path("/proc").glob("[0-9]*/stat"):
    try:
     text=p.read_text();name=text.split("(",1)[1].rsplit(")",1)[0]
     if name.startswith(("ksoftirqd/","iou-wrk","kworker/")):out["visible_kernel_workers"][p.parent.name]=dict(stat=text,status=read(p.parent/"status"),schedstat=read(p.parent/"schedstat"))
    except (OSError,ValueError):pass
  out["sample_end_ns"]=time.monotonic_ns()
  emit("sample",**out)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 emit("observer_metadata",pid=args.pid,cgroup=str(cg),clock_tick=os.sysconf("SC_CLK_TCK"),boot=read("/proc/sys/kernel/random/boot_id"),time_namespace=read("/proc/self/timens_offsets"),
      maps=map_metadata(),kernel=read("/proc/sys/kernel/osrelease"),cadence_s=.1,thread_detail_cadence_s=1,source=Path(__file__).read_text(),scope="Host/device totals include all tenants; process/group/cgroup nested, never summed. GPU calls occur only at observation boundaries; interval device={} is unavailable, not no GPU activity.")
 sample(True,True)
 try:
  while not done:
   time.sleep(.1);count+=1;sample(count%10==0)
 finally:
  sample(True,True);emit("observer_terminal",samples=count+2,observer_cpu_ns=time.process_time_ns());raw.close();nv.nvmlShutdown()
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--pid",type=int,required=True);ap.add_argument("--output",required=True);main(ap.parse_args())
