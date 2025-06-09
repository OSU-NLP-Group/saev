

```
[I] samuelstevens@a0001 ~/p/saev (main)> fio --name=net --filename=/fs/scratch/PAS2136/samuelstevens/cache/saev/366017a10220b85014ae0a594276b25f6ea3d756b74d1d3218da1e34ffcf32e9/acts000000.bin --rw=read --bs=1M --direct=1 --iodepth=16 --runtime=30 --time_based
net: (g=0): rw=read, bs=(R) 1024KiB-1024KiB, (W) 1024KiB-1024KiB, (T) 1024KiB-1024KiB, ioengine=psync, iodepth=16
fio-3.35
Starting 1 process
note: both iodepth >= 1 and synchronous I/O engine are selected, queue depth will be capped at 1
Jobs: 1 (f=1): [R(1)][100.0%][r=550MiB/s][r=550 IOPS][eta 00m:00s]
net: (groupid=0, jobs=1): err= 0: pid=3555835: Sun Jun  8 22:34:38 2025
  read: IOPS=796, BW=796MiB/s (835MB/s)(23.3GiB/30001msec)
    clat (usec): min=334, max=14096, avg=1255.50, stdev=747.84
     lat (usec): min=334, max=14096, avg=1255.54, stdev=747.84
    clat percentiles (usec):
     |  1.00th=[  383],  5.00th=[  412], 10.00th=[  420], 20.00th=[  676],
     | 30.00th=[  914], 40.00th=[  922], 50.00th=[  979], 60.00th=[ 1500],
     | 70.00th=[ 1614], 80.00th=[ 1680], 90.00th=[ 2057], 95.00th=[ 2638],
     | 99.00th=[ 3818], 99.50th=[ 4359], 99.90th=[ 5800], 99.95th=[ 6259],
     | 99.99th=[ 9110]
   bw (  KiB/s): min=434176, max=2119680, per=100.00%, avg=820380.20, stdev=403858.41, samples=59
   iops        : min=  424, max= 2070, avg=801.15, stdev=394.39, samples=59
  lat (usec)   : 500=16.82%, 750=8.74%, 1000=27.34%
  lat (msec)   : 2=36.48%, 4=9.84%, 10=0.77%, 20=0.01%
  cpu          : usr=0.06%, sys=2.67%, ctx=23995, majf=0, minf=266
  IO depths    : 1=100.0%, 2=0.0%, 4=0.0%, 8=0.0%, 16=0.0%, 32=0.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     issued rwts: total=23888,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=16

Run status group 0 (all jobs):
   READ: bw=796MiB/s (835MB/s), 796MiB/s-796MiB/s (835MB/s-835MB/s), io=23.3GiB (25.0GB), run=30001-30001msec
[I] samuelstevens@a0001 ~/p/saev (main)> fio --name=net --filename=/fs/scratch/PAS2136/samuelstevens/cache/saev/366017a10220b85014ae0a594276b25f6ea3d756b74d1d3218da1e34ffcf32e9/acts000000.bin --rw=randread --bs=4K --direct=1 --iodepth=16 --runtime=30 --time_based
net: (g=0): rw=randread, bs=(R) 4096B-4096B, (W) 4096B-4096B, (T) 4096B-4096B, ioengine=psync, iodepth=16
fio-3.35
Starting 1 process
note: both iodepth >= 1 and synchronous I/O engine are selected, queue depth will be capped at 1
Jobs: 1 (f=1): [r(1)][100.0%][r=21.4MiB/s][r=5477 IOPS][eta 00m:00s]
net: (groupid=0, jobs=1): err= 0: pid=3557436: Sun Jun  8 22:50:52 2025
  read: IOPS=5865, BW=22.9MiB/s (24.0MB/s)(687MiB/30001msec)
    clat (usec): min=68, max=13344, avg=170.06, stdev=206.09
     lat (usec): min=68, max=13344, avg=170.10, stdev=206.09
    clat percentiles (usec):
     |  1.00th=[   76],  5.00th=[   79], 10.00th=[   82], 20.00th=[   90],
     | 30.00th=[   97], 40.00th=[  103], 50.00th=[  108], 60.00th=[  115],
     | 70.00th=[  206], 80.00th=[  229], 90.00th=[  260], 95.00th=[  306],
     | 99.00th=[ 1139], 99.50th=[ 1598], 99.90th=[ 2442], 99.95th=[ 2769],
     | 99.99th=[ 3589]
   bw (  KiB/s): min=11240, max=34792, per=100.00%, avg=23509.15, stdev=4943.22, samples=59
   iops        : min= 2810, max= 8698, avg=5877.29, stdev=1235.81, samples=59
  lat (usec)   : 100=34.78%, 250=53.34%, 500=9.21%, 750=0.92%, 1000=0.54%
  lat (msec)   : 2=0.98%, 4=0.23%, 10=0.01%, 20=0.01%
  cpu          : usr=0.46%, sys=8.91%, ctx=176070, majf=0, minf=10
  IO depths    : 1=100.0%, 2=0.0%, 4=0.0%, 8=0.0%, 16=0.0%, 32=0.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     issued rwts: total=175978,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=16

Run status group 0 (all jobs):
   READ: bw=22.9MiB/s (24.0MB/s), 22.9MiB/s-22.9MiB/s (24.0MB/s-24.0MB/s), io=687MiB (721MB), run=30001-30001msec
[I] samuelstevens@a0001 ~/p/saev (main)>
```
