[xpleva07@acn01.karolina BUT_FIT_PCG_1]$ make run
cd build && \
./nbodyCpu 81920 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
./nbody0 81920 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
./nbody1 81920 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 && \
./nbody2 81920 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5 
       NBODY CPU simulation
N:                       81920
dt:                      0.010000
steps:                   100
Time: 352.768341 s
Reference center of mass: 50.083378, 49.872196, 49.970253, 102133114339328.000000
Center of mass on CPU: 50.083332, 49.872036, 49.970482, 102133642821632.000000
       NBODY GPU simulation
N:                       81920
dt:                      0.010000
steps:                   100
threads/block:           512
blocks/grid:             160
reduction threads/block: 128
reduction blocks/grid:   16
Time: 11.314473 s
Reference center of mass: 50.083378, 49.872196, 49.970249, 102133114339328.000000
Center of mass on GPU: 50.083344, 49.872066, 49.970478, 102133626044416.000000
       NBODY GPU simulation
N:                       81920
dt:                      0.010000
steps:                   100
threads/block:           512
blocks/grid:             160
reduction threads/block: 128
reduction blocks/grid:   16
Time: 7.108525 s
Reference center of mass: 50.082958, 49.876953, 49.969730, 102133114339328.000000
Center of mass on GPU: 50.082932, 49.877113, 49.969677, 102133651210240.000000
       NBODY GPU simulation
N:                       81920
dt:                      0.010000
steps:                   100
threads/block:           512
blocks/grid:             160
reduction threads/block: 128
reduction blocks/grid:   16
Time: 0.258499 s
Reference center of mass: 50.083046, 49.877125, 49.969864, 102133114339328.000000
Center of mass on GPU: 50.082985, 49.877151, 49.969704, 102133735096320.000000
