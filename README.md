# GPU simulation of N-body system using CUDA
### project1 to BUT FIT PCG

Simple simulator of N-Body system implemented in CUDA. Contains 3 steps of implementation

### Step0
Raw rewrite CPU version to CUDA

### Step1
Merge 3 kernels of compution (claculate speed, claculate colision and update particles) in one kenel.

### Step2
Use shared memory to minimalize access to global GPU memory

### Step3
Use reduction on warplevel (using reduction tree) in calucation of center of mass function

### Step4 (final)
Use streams to overwrap calculation on GPU and CPU -> GPU and GPU -> CPU data transfers

## Assigment

in `doc.pdf`

## Build

On karolina supercomputer first load modules using `. loadModules.sh` then build using `make`

## Using

after build is in `build` dir builded binares you can simply run binares of any step.

```sh
# ./nbodyN <NUM OF PARTECLES> <STEP> <STEPS> <THREATS> <WRITE INTENSITY> <REDUCTION THREADS> <REDUCTION THREADS PER BLOCK> <INPUT .h5> <output .h5>
./nbody4 4096 0.01f 100 512 5 2048 128 input.h5 outputCpu.h5
```

input `.h5` can by simple generated by `./gen`

```sh
# ./gen <NUM OF PARTICLES> <FILE>
./gen 4096 input.h5
```

## Results
Final results is in file `nbody.md`