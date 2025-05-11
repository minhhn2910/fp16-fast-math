#!/usr/bin/env bash
# Auto-detect GPU arch, compile four impls, sweep functions, write BENCHMARK_REPORT.md
#
#   ./run_auto_bench.sh
#   ./run_auto_bench.sh --arch sm_90
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
NVCC=${NVCC:-nvcc}
NVCC_ARCH=${NVCC_ARCH:-native}
CUDA_DEVICE=${CUDA_DEVICE:-0}
OUT_DIR="${SCRIPT_DIR}/results"
FUNCS=(rcp exp log log2 log10 sqrt rsqrt sin cos asin acos)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR=$2; shift 2 ;;
    --arch) NVCC_ARCH=$2; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--arch sm_XX] [--out DIR]"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}" "${SCRIPT_DIR}/build"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

GPU_NAME_RAW=$(nvidia-smi --id="${CUDA_DEVICE}" --query-gpu=name --format=csv,noheader | head -n1)
GPU_NAME=$(echo "${GPU_NAME_RAW}" | sed 's/ /_/g')
SM_CAP=$(nvidia-smi --id="${CUDA_DEVICE}" --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
NVCC_VER=$("${NVCC}" --version | grep -o 'release [^,]*' | head -n1)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SAFE_SM=${SM_CAP/./}

echo "GPU ${GPU_NAME_RAW}  SM ${SM_CAP}  ${NVCC_VER}  -arch=${NVCC_ARCH}"

FLAGS=(-O3 -std=c++17 -arch="${NVCC_ARCH}" -I"${ROOT_DIR}/include" --diag-suppress 177,550 -Xptxas=-v)

compile_one() {
  local name=$1
  shift
  echo "compiling ${name}"
  local log="${SCRIPT_DIR}/build/${name}.ptxas"
  if ! "${NVCC}" "${FLAGS[@]}" "$@" "${SCRIPT_DIR}/bench.cu" \
      -o "${SCRIPT_DIR}/build/${name}" > "${log}" 2>&1; then
    cat "${log}" >&2
    exit 1
  fi
}

compile_one bench_lib      -DBENCH_IMPL=0
compile_one bench_cuda_h2  -DBENCH_IMPL=1
compile_one bench_f32      -DBENCH_IMPL=2
compile_one bench_f32_fast -DBENCH_IMPL=2 -DBENCH_FAST_MATH -use_fast_math

if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump -sass "${SCRIPT_DIR}/build/bench_lib" > "${SCRIPT_DIR}/build/bench_lib.sass"
else
  echo "warning: cuobjdump not found; SASS table will be omitted" >&2
  : > "${SCRIPT_DIR}/build/bench_lib.sass"
fi

CSV="${OUT_DIR}/bench_${GPU_NAME}_sm${SAFE_SM}_${STAMP}.csv"
{
  echo "# gpu=${GPU_NAME_RAW}"
  echo "# sm=${SM_CAP}"
  echo "# nvcc=${NVCC_VER}"
  echo "# nvcc_arch=${NVCC_ARCH}"
  echo "# timestamp=${STAMP}"
  echo "impl,func,rel_err,gelems,cycles"
} > "${CSV}"

for func in "${FUNCS[@]}"; do
  echo "  ${func}"
  for bin in bench_lib bench_cuda_h2 bench_f32 bench_f32_fast; do
    "${SCRIPT_DIR}/build/${bin}" --func "${func}" --csv >> "${CSV}"
  done
done

ln -sfr "${CSV}" "${OUT_DIR}/latest.csv"

python3 "${SCRIPT_DIR}/generate_report.py" \
  --csv "${CSV}" \
  --sass "${SCRIPT_DIR}/build/bench_lib.sass" \
  --ptxas "${SCRIPT_DIR}/build/bench_lib.ptxas" \
  --plot "${OUT_DIR}/rel_error.png" \
  --out "${OUT_DIR}/bench_${GPU_NAME}_sm${SAFE_SM}_${STAMP}.md" \
  --root-report "${ROOT_DIR}/BENCHMARK_REPORT.md"

echo "wrote ${ROOT_DIR}/BENCHMARK_REPORT.md"
