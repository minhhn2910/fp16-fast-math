## CUDA FP16x2 approximate math library
### Implementation:
  The folder `/include` has all necessary file to use our approximate math library.
  We implemented most popular math functions with the same function signature as cuda `math.h`. Thus,
  the library can be used by simply including our header file `./include/fast_math.cuh`
### Speedup and Error Rate:

<img width="835" height="464" alt="fp16_fast_math_lib" src="https://github.com/user-attachments/assets/c97fe2ca-9f9d-4f58-8f86-5c1dc89bde01" />

The figure above is from Pascal / Volta. 
On those GPUs the SFU has no native `half2` special functions; packed `h2sin` / `h2exp` / … go through fp32.
```
LDG.E.U16 R2, [R2]; //load from memory
F2F.F32.F16 R0, R2; //promoted to float
....
RRO.EX2 R4, R0; //range reduction
MUFU.EX2 R0, R4; //compute exp()
....
F2F.F16.F32 R0, R0; //convert back to half
....
STG.E.U16 [R2], R0; //store result to memory
```
Hardware `half2` in the SFU appears from Turing onward.
Updated numbers for Ampere are in **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**.

### Supporting operation overload:
  Because some of the older versions of CUDA do not support operator overload for half2 type, the header `./include/half2_operator_overload.cuh` will help our code compiled successfully.
  Noted that some newer versions of CUDA does support most of the operator overload defined in the above header file. In these cases, we need to remove the conflict definitions from `half2_operator_overload.cuh` to avoid compiler complain.
  The current half2_operator_overload may receives complain from some earlier or later CUDA than our test version `CUDA 10.0`. If it does not work, feel free to open issue in this repository, I will try my best to help.
### Microbenchmarking the approx math library

  **New architecture (auto-detect):** `./microbenchmarkSFU/auto_bench/` compiles for the current GPU (`-arch=native`, or `--arch sm_XX`), sweeps all functions, and writes **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**.

  ```bash
  cd microbenchmarkSFU/auto_bench
  ./run_auto_bench.sh
  ./run_auto_bench.sh --arch sm_90
  ```

  **Original V100-era scripts** are in `./microbenchmarkSFU/benchmarks/` (`compile.sh` is hardcoded to `sm_70`):
  1. check `mathfloat.cu` and `mathhalf2.cu` to define the desired function to benchmark e.g. : `#define HOST_FUNC asin` and `#define DEV_FUNC fast_h2asin`
  2. Compile using the fast math library `./compile.sh` and run test `./run.sh`
  3. Compile using the default math library `./compile_float_fast.sh` and run test `./run.sh`
  4. If you wish to compare with CUDA fp16 math library, change the `#include "../../include/fast_math.cuh"` in `mathhalf2.cu` to `#include "../../include/cuda_math.cuh"`. OR you can just simply remove the include statement.
### References:
It has been used as a contribution in our research paper:

Ho, Nhut-Minh, Himeshi De silva, and Weng-Fai Wong. "GRAM: A framework for dynamically mixing precisions in GPU applications." ACM Transactions on Architecture and Code Optimization (TACO) 18.2 (2021): 1-24.
