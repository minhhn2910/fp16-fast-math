## CUDA FP16x2 approximate math library
### Implementation:
  The folder `/include` has all necessary file to use our approximate math library.
  We implemented most popular math functions with the same function signature as cuda `math.h`. Thus,
  the library can be used by simply including our header file `./include/fast_math.cuh`
### Supporting operation overload:
  Because some of the older versions of CUDA do not support operator overload for half2 type, the header `./include/half2_operator_overload.cuh` will help our code compiled successfully.
  Noted that some newer versions of CUDA does support most of the operator overload defined in the above header file. In these cases, we need to remove the conflict definitions from `half2_operator_overload.cuh` to avoid compiler complain.
  The current half2_operator_overload may receives complain from some earlier or later CUDA than our test version `CUDA 10.0`. If it does not work, feel free to open issue in this repository, I will try my best to help.
### Microbenchmarking the approx math library
  The folder `./microbenchmarkSFU/benchmarks/` contains the simple micro benchmark to measure accuracy and performance of various default math function in CUDA and our approximate equivalence.
  To run a microbenchmark for comparison:
  1. check `mathfloat.cu` and `mathhalf2.cu` to define the desired function to benchmark e.g. : `#define HOST_FUNC asin` and `#define DEV_FUNC fast_h2asin`
  2. Compile using the fast math library `./compile.sh` and run test `./run.sh`
  3. Compile using the default math library `./compile_float_fast.sh` and run test `./run.sh`
  4. If you wish to compare with CUDA fp16 math library, change the `#include "../../include/fast_math.cuh"` in `mathhalf2.cu` to `#include "../../include/cuda_math.cuh"`. OR you can just simply remove the include statement.
