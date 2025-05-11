// Microbenchmark: mean rel-error vs fp32, 8-chain throughput, clock64 latency.
// -DBENCH_IMPL=0 lib | 1 cuda_h2 | 2 f32    add -DBENCH_FAST_MATH with -use_fast_math

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef BENCH_IMPL
#define BENCH_IMPL 0
#endif

#if BENCH_IMPL == 1
#include "cuda_math.cuh"
#elif BENCH_IMPL == 0
#include "fast_math.cuh"
#endif

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = (stmt);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s (%s:%d)\n", cudaGetErrorString(err),      \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

enum Func : int {
  RCP, EXP, LOG, LOG2, LOG10, SQRT, RSQRT, SIN, COS, ASIN, ACOS, FUNC_COUNT
};

static const char *kName[] = {"rcp",  "exp", "log",  "log2", "log10", "sqrt",
                              "rsqrt", "sin", "cos",  "asin", "acos"};

struct Range {
  float lo, hi;
};

static const Range kDomain[] = {
    {0.05f, 16.f}, {-8.f, 8.f},   {1e-3f, 16.f}, {1e-3f, 16.f}, {1e-3f, 16.f},
    {0.f, 16.f},   {1e-3f, 16.f}, {-8.f, 8.f},   {-8.f, 8.f},   {-1.f, 1.f},
    {-1.f, 1.f},
};

static float rcp_f(float x) { return 1.f / x; }
static float rsqrt_f(float x) { return 1.f / sqrtf(x); }
static float (*const kRefF32[])(float) = {rcp_f,  expf,  logf,  log2f, log10f,
                                          sqrtf, rsqrt_f, sinf,  cosf,  asinf,
                                          acosf};

constexpr int kNAcc = 1 << 20;
constexpr int kNTput = 1 << 20;
constexpr int kTputLoop = 64;
constexpr int kLatN = 32;
constexpr int kLatLoop = 128;
constexpr int kWarmup = 3;
constexpr int kRepeats = 7;
constexpr int kBlock = 256;
constexpr int kChains = 8;

#if BENCH_IMPL != 2
using Vec = half2;
using Lane = __half;
constexpr int kLanes = 2;
#else
using Vec = float;
using Lane = float;
constexpr int kLanes = 1;
#endif

template <int F>
__device__ __forceinline__ Vec eval(Vec x) {
#if BENCH_IMPL != 2
  if constexpr (F == RCP)
    return fast_h2rcp(x);
  if constexpr (F == EXP)
    return fast_h2exp(x);
  if constexpr (F == LOG)
    return fast_h2log(x);
  if constexpr (F == LOG2)
    return fastest_h2log2(x);
  if constexpr (F == LOG10)
    return fast_h2log10(x);
  if constexpr (F == SQRT)
    return fast_h2sqrt(x);
  if constexpr (F == RSQRT)
    return fast_h2rsqrt(x);
  if constexpr (F == SIN)
    return fast_h2sin(x);
  if constexpr (F == COS)
    return fast_h2cos(x);
  if constexpr (F == ASIN)
    return fast_h2asin(x);
  if constexpr (F == ACOS)
    return fast_h2acos(x);
#else
  if constexpr (F == RCP)
    return 1.f / x;
  if constexpr (F == EXP)
    return expf(x);
  if constexpr (F == LOG)
    return logf(x);
  if constexpr (F == LOG2)
    return log2f(x);
  if constexpr (F == LOG10)
    return log10f(x);
  if constexpr (F == SQRT)
    return sqrtf(x);
  if constexpr (F == RSQRT)
    return rsqrtf(x);
  if constexpr (F == SIN)
    return sinf(x);
  if constexpr (F == COS)
    return cosf(x);
  if constexpr (F == ASIN)
    return asinf(x);
  if constexpr (F == ACOS)
    return acosf(x);
#endif
  return x;
}

__device__ __forceinline__ long long sm_clock() {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory");
  return t;
}

template <int F>
__global__ void k_acc(const Vec *in, Vec *out, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n)
    out[tid] = eval<F>(in[tid]);
}

template <int F>
__global__ void k_tput(const Vec *in, Vec *out, int n, int loop, Vec inc) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n)
    return;
  Vec a = in[tid];
  Vec b = a + inc, c = b + inc, d = c + inc;
  Vec e = d + inc, f = e + inc, g = f + inc, h = g + inc;
#pragma unroll 1
  for (int i = 0; i < loop; i++) {
    a = eval<F>(a);
    b = eval<F>(b);
    c = eval<F>(c);
    d = eval<F>(d);
    e = eval<F>(e);
    f = eval<F>(f);
    g = eval<F>(g);
    h = eval<F>(h);
  }
  out[tid] = a + b + c + d + e + f + g + h;
}

template <int F>
__global__ void k_lat(const Vec *in, Vec *out, unsigned long long *t0,
                      unsigned long long *t1, int n, int loop) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n)
    return;
  Vec x = eval<F>(in[tid]);
  auto start = (unsigned long long)sm_clock();
#pragma unroll 1
  for (int i = 0; i < loop; i++)
    x = eval<F>(x);
  auto stop = (unsigned long long)sm_clock();
  t0[tid] = start;
  t1[tid] = stop;
  out[tid] = x;
}

#define DISPATCH(K, ...)                                                       \
  switch (func) {                                                              \
  case RCP:    K<RCP><<<grid, block>>>(__VA_ARGS__); break;                    \
  case EXP:    K<EXP><<<grid, block>>>(__VA_ARGS__); break;                    \
  case LOG:    K<LOG><<<grid, block>>>(__VA_ARGS__); break;                    \
  case LOG2:   K<LOG2><<<grid, block>>>(__VA_ARGS__); break;                   \
  case LOG10:  K<LOG10><<<grid, block>>>(__VA_ARGS__); break;                  \
  case SQRT:   K<SQRT><<<grid, block>>>(__VA_ARGS__); break;                   \
  case RSQRT:  K<RSQRT><<<grid, block>>>(__VA_ARGS__); break;                  \
  case SIN:    K<SIN><<<grid, block>>>(__VA_ARGS__); break;                    \
  case COS:    K<COS><<<grid, block>>>(__VA_ARGS__); break;                    \
  case ASIN:   K<ASIN><<<grid, block>>>(__VA_ARGS__); break;                   \
  case ACOS:   K<ACOS><<<grid, block>>>(__VA_ARGS__); break;                   \
  }

template <typename T>
struct Dev {
  T *p{};
  explicit Dev(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev &) = delete;
};

static const char *impl_name() {
#ifdef BENCH_FAST_MATH
  return "f32_fast";
#elif BENCH_IMPL == 1
  return "cuda_h2";
#elif BENCH_IMPL == 2
  return "f32";
#else
  return "lib";
#endif
}

static int parse_func(const char *s) {
  for (int i = 0; i < FUNC_COUNT; i++)
    if (std::strcmp(s, kName[i]) == 0)
      return i;
  return -1;
}

static std::vector<double> sample(int n, Range r) {
  std::vector<double> x(n);
  for (int i = 0; i < n; i++)
    x[i] = r.lo + (r.hi - r.lo) * (std::rand() / (double)RAND_MAX);
  return x;
}

static std::vector<Lane> quantize(const std::vector<double> &x) {
  std::vector<Lane> h(x.size());
  for (size_t i = 0; i < x.size(); i++)
#if BENCH_IMPL != 2
    h[i] = __float2half_rn((float)x[i]);
#else
    h[i] = (float)x[i];
#endif
  return h;
}

static float to_f32(Lane s) {
#if BENCH_IMPL != 2
  return __half2float(s);
#else
  return s;
#endif
}

static double mean_rel_err(int func, const std::vector<double> &x,
                           const std::vector<float> &y) {
  double sum = 0;
  long n = 0;
  for (size_t i = 0; i < x.size(); i++) {
    float ref = kRefF32[func]((float)x[i]);
    if (!std::isfinite(y[i]) || !std::isfinite(ref) || std::fabs(ref) <= 1e-4f)
      continue;
    double r = std::fabs((double)y[i] - ref) / std::fabs(ref);
    sum += r > 1.0 ? 1.0 : r;
    n++;
  }
  return n ? sum / n : 0.0;
}

static Vec make_inc() {
#if BENCH_IMPL != 2
  return __float2half2_rn(0.01f);
#else
  return 0.01f;
#endif
}

static double run_accuracy(int func, Range r) {
  auto x = sample(kNAcc, r);
  auto h = quantize(x);
  for (size_t i = 0; i < x.size(); i++)
    x[i] = to_f32(h[i]);

  const int n = (int)h.size() / kLanes;
  Dev<Vec> in(n), out(n);
  CUDA_CHECK(cudaMemcpy(in.p, h.data(), h.size() * sizeof(Lane),
                        cudaMemcpyHostToDevice));
  int grid = (n + kBlock - 1) / kBlock, block = kBlock;
  DISPATCH(k_acc, in.p, out.p, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Lane> hout(h.size());
  CUDA_CHECK(cudaMemcpy(hout.data(), out.p, h.size() * sizeof(Lane),
                        cudaMemcpyDeviceToHost));
  std::vector<float> y(h.size());
  for (size_t i = 0; i < y.size(); i++)
    y[i] = to_f32(hout[i]);
  return mean_rel_err(func, x, y);
}

static double run_throughput(int func, Range r) {
  auto h = quantize(sample(kNTput, r));
  const int n = (int)h.size() / kLanes;
  Dev<Vec> in(n), out(n);
  CUDA_CHECK(cudaMemcpy(in.p, h.data(), h.size() * sizeof(Lane),
                        cudaMemcpyHostToDevice));
  int grid = (n + kBlock - 1) / kBlock, block = kBlock;
  Vec inc = make_inc();
  auto launch = [&] {
    DISPATCH(k_tput, in.p, out.p, n, kTputLoop, inc);
  };
  for (int i = 0; i < kWarmup; i++) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float acc = 0;
  for (int i = 0; i < kRepeats; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    acc += ms;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  double ms = acc / kRepeats;
  return (double)n * kLanes * kChains * kTputLoop / (ms * 1e6);
}

static double run_latency(int func, Range r) {
  auto h = quantize(sample(kLatN * kLanes, r));
  const int n = kLatN;
  Dev<Vec> in(n), out(n);
  Dev<unsigned long long> t0(n), t1(n);
  CUDA_CHECK(cudaMemcpy(in.p, h.data(), h.size() * sizeof(Lane),
                        cudaMemcpyHostToDevice));
  int grid = 1, block = n;
  auto launch = [&] {
    DISPATCH(k_lat, in.p, out.p, t0.p, t1.p, n, kLatLoop);
  };
  for (int i = 0; i < kWarmup; i++) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<unsigned long long> s(n), e(n);
  CUDA_CHECK(cudaMemcpy(s.data(), t0.p, n * sizeof(*t0.p),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(e.data(), t1.p, n * sizeof(*t1.p),
                        cudaMemcpyDeviceToHost));
  double sum = 0;
  for (int i = 0; i < n; i++)
    sum += (double)(e[i] - s[i]) / kLatLoop;
  return sum / n;
}

int main(int argc, char **argv) {
  int func = -1;
  bool csv = false;
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--func") && i + 1 < argc)
      func = parse_func(argv[++i]);
    else if (!std::strcmp(argv[i], "--csv"))
      csv = true;
    else if (!std::strcmp(argv[i], "--list")) {
      for (int f = 0; f < FUNC_COUNT; f++)
        printf("%s\n", kName[f]);
      return 0;
    }
  }
  if (func < 0) {
    fprintf(stderr, "usage: %s --func <name> [--csv]\n", argv[0]);
    return 1;
  }

  std::srand(1234);
  Range r = kDomain[func];
  double rel = run_accuracy(func, r);
  double gelems = run_throughput(func, r);
  double cycles = run_latency(func, r);

  if (csv)
    printf("%s,%s,%.6g,%.4f,%.2f\n", impl_name(), kName[func], rel, gelems,
           cycles);
  else
    printf("%s %s  rel_err(vs fp32)=%.4g  tput=%.2f Gelem/s  latency=%.1f cyc\n",
           impl_name(), kName[func], rel, gelems, cycles);
  return 0;
}
