#!/usr/bin/env python3
"""Build a short benchmark report with a grouped rel-error plot."""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, NullLocator


FUNCS = [
    "rcp", "exp", "log", "log2", "log10", "sqrt",
    "rsqrt", "sin", "cos", "asin", "acos",
]


def parse_csv(path):
    meta, rows = {}, []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].strip().split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            if header is None:
                header = line
                continue
            rows.append(next(csv.DictReader([header, line])))
    by = {(r["impl"], r["func"]): r for r in rows}
    return meta, by


def ffmt(x, spec):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return format(v, spec)


def speedup(lib_g, base_g):
    try:
        a, b = float(lib_g), float(base_g)
    except (TypeError, ValueError):
        return "—"
    if b <= 0:
        return "—"
    return f"{a / b:.2f}×"


def parse_ptxas(path):
    if not path or not os.path.isfile(path):
        return {}
    text = open(path, errors="replace").read()
    regs = {}
    current = None
    for line in text.splitlines():
        m = re.search(r"(?:k_tput|tput_h2)ILi(\d+)E", line)
        if m:
            current = int(m.group(1))
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m and current is not None:
            if 0 <= current < len(FUNCS):
                regs[FUNCS[current]] = int(m.group(1))
            current = None
    return regs


def classify_sass(mnemonic):
    m = mnemonic.upper()
    if m.startswith("MUFU"):
        return "MUFU"
    if m.startswith(("HFMA2", "HADD2", "HMUL2", "HSET2", "HSETP2", "HMNMX2")):
        return "HFMA2"
    if m.startswith(("LOP3", "LOP.", "PLOP3", "XOR", "AND", "OR", "PRMT")):
        return "LOP3"
    if m.startswith(("I2F", "F2I", "F2F", "I2I", "I2IP", "FRND", "CVT")):
        return "CVT"
    return None


def parse_sass(path):
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return {}
    text = open(path, errors="replace").read()
    mix = defaultdict(lambda: {"MUFU": 0, "HFMA2": 0, "LOP3": 0, "CVT": 0})
    func_id = None
    for line in text.splitlines():
        m = re.search(r"Function\s*:\s*\S*(?:k_lat|lat_h2)ILi(\d+)E", line)
        if m:
            func_id = int(m.group(1))
            continue
        if line.strip().startswith("Function"):
            func_id = None
            continue
        if func_id is None or not (0 <= func_id < len(FUNCS)):
            continue
        inst = re.search(
            r"/\*[0-9a-fA-F]+\*/\s+(?:@[!\w]+\s+)?([A-Z][A-Za-z0-9.]*)", line
        )
        if not inst:
            continue
        kind = classify_sass(inst.group(1))
        if kind:
            mix[FUNCS[func_id]][kind] += 1
    return mix


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_rel_error(by, out_path, gpu, sm):
    lib = [float(by.get(("lib", func), {}).get("rel_err", "nan")) for func in FUNCS]

    x = np.arange(len(FUNCS))
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.bar(x, lib, 0.62, color="#3d4f66")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e0)
    yticks = [1e-5, 1e-3, 1e-1, 1e0]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(x, FUNCS)
    ax.set_ylabel("Mean relative error vs fp32")
    ax.set_title("Average relative error vs fp32")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.text(
        0.01,
        0.02,
        f"Source: auto_bench · {gpu} SM {sm} · mean |y − f32(x)| / |f32(x)|",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_report(meta, by, sass, regs, plot_href):
    gpu = meta.get("gpu", "GPU")
    sm = meta.get("sm", "?")
    nvcc = meta.get("nvcc", "")
    arch = meta.get("nvcc_arch", "native")

    perf_rows = []
    sass_rows = []
    for func in FUNCS:
        lib = by.get(("lib", func), {})
        h2 = by.get(("cuda_h2", func), {})
        f32 = by.get(("f32", func), {})
        ff = by.get(("f32_fast", func), {})
        perf_rows.append(
            [
                func,
                ffmt(lib.get("gelems"), ".0f"),
                speedup(lib.get("gelems"), h2.get("gelems")),
                speedup(lib.get("gelems"), ff.get("gelems")),
                speedup(lib.get("gelems"), f32.get("gelems")),
                ffmt(lib.get("cycles"), ".0f"),
            ]
        )
        mix = sass.get(func, {})
        nreg = regs.get(func)
        alu = (
            mix.get("HFMA2", 0) + mix.get("LOP3", 0) + mix.get("CVT", 0)
            if mix
            else None
        )
        sass_rows.append(
            [
                func,
                str(alu) if alu is not None else "—",
                str(mix.get("MUFU", 0) if mix else "—"),
                str(nreg) if nreg is not None else "—",
            ]
        )

    sass_section = f"""
## SASS (this library)

`cuobjdump -sass` on the latency kernel. **ALU** = packed fp16 math + integer
logic + converts (software path). **MUFU** = hardware special-function unit.
Regs = throughput kernel.

{md_table(["Function", "ALU", "MUFU", "Regs"], sass_rows)}
"""

    return f"""# FP16 fast-math report

{gpu}, SM {sm}, {nvcc}, `-arch={arch}`.

## Method

- Four implementations: this library (`half2`), CUDA `h2*`, fp32, and fp32 compiled with `-use_fast_math`.
- Accuracy: mean relative error vs host fp32, `|y − f32(x)| / |f32(x)|` (log axis 10⁻⁵ … 10⁰, step 10²).
- Throughput: 8 independent chains per thread; GElem/s counts scalar results. Speedup is library throughput ÷ baseline.
- Latency: one dependent chain; `clock64()` cycles per call.
- SASS: instruction mix of the latency kernel. This library is ALU-only (no MUFU).

## Accuracy

![Mean relative error vs fp32]({plot_href})

## Performance

{md_table(["Function", "GElem/s", "vs h2", "vs fast-math", "vs fp32", "Cycles"], perf_rows)}
{sass_section}
## Rerun

```bash
cd microbenchmarkSFU/auto_bench
./run_auto_bench.sh
./run_auto_bench.sh --arch sm_90
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-report", default="")
    ap.add_argument("--sass", default="")
    ap.add_argument("--ptxas", default="")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    meta, by = parse_csv(args.csv)
    sass = parse_sass(args.sass)
    regs = parse_ptxas(args.ptxas)
    plot_path = args.plot or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)), "rel_error.png"
    )
    plot_rel_error(by, plot_path, meta.get("gpu", ""), meta.get("sm", ""))

    results_md = build_report(meta, by, sass, regs, os.path.basename(plot_path))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(results_md)
    if args.root_report:
        rel = os.path.relpath(
            os.path.abspath(plot_path),
            os.path.dirname(os.path.abspath(args.root_report)),
        )
        with open(args.root_report, "w") as f:
            f.write(build_report(meta, by, sass, regs, rel))


if __name__ == "__main__":
    main()
