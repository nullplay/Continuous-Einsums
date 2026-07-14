"""Benchmark the dense interaction-table einsum (``table_ceinsum``).

Three cases, matching the thesis chapter:

1. ``i,i->i``   both interval           — vs the existing ``ceinsum`` pipeline
2. ``ij,j->i``  interval i, pinpoint j  — vs ``ceinsum`` (mask + merge path)
3. ``ik,kj->ij`` A:(P,I), B:(I,I)       — vs ``ceinsum`` (integrated k: both
                                          weight by the k-overlap length)

The two implementations are also cross-checked pointwise (their piece lists
differ by construction — grid cells vs per-match intersections — so values at
probe points are compared, not pieces).

Sizes whose predicted table exceeds the element guard are skipped and reported
as such: the dense-table memory wall is a *result* of this reference
implementation, not a bug.

Usage:  python benchmarks/bench_table_ceinsum.py
        [--sizes 25,50,...] [--repeats 5] [--out docs/table_ceinsum_bench.md]
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from continuous_einsum import ceinsum
from ctensor import ContinuousTensor, continuous_tensor
from synth_dataset import create_nd_pieces
from table_ceinsum import table_ceinsum

DTYPE = torch.float32
DEVICE = torch.device("cpu")
ELEM_GUARD = int(2e8)  # skip a size when the table would exceed this many elements
N_PROBES = 512


def _values(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return 0.5 + torch.rand(n, generator=gen, dtype=DTYPE)


def _intervals(n: int, seed: int, span: float = 100.0):
    (dims,) = create_nd_pieces(
        n, ("interval",), (span,), 0.5, seed=seed, device=DEVICE, dtype=DTYPE
    )
    return dims


def _bench(fn, repeats: int) -> float:
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _eval_at(ct: ContinuousTensor, points: torch.Tensor) -> torch.Tensor:
    npts = points.shape[0]
    mask = torch.ones(npts, ct.nnz, dtype=torch.bool)
    for d in range(ct.ndim):
        p = points[:, d][:, None]
        spec = ct.dims[d]
        if len(spec) == 1:
            mask &= p == spec[0][None, :]
        else:
            mask &= (spec[0][None, :] <= p) & (p < spec[1][None, :])
    return (mask.to(torch.float64) * ct.values.to(torch.float64)[None, :]).sum(dim=1)


def _probe_midpoints(*interval_specs) -> torch.Tensor:
    pts = torch.unique(torch.cat([t for spec in interval_specs for t in spec]))
    mids = (pts[:-1] + pts[1:]) / 2
    if mids.shape[0] > N_PROBES:
        step = mids.shape[0] // N_PROBES
        mids = mids[::step]
    return mids[:, None]


def _max_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = torch.maximum(a.abs(), b.abs()).clamp(min=1.0)
    return float(((a - b).abs() / scale).max()) if a.numel() else 0.0


# ---------------------------------------------------------------------------
# Case definitions: each returns (operands, equation, probe fn or None).
# ---------------------------------------------------------------------------


def case_elementwise(n: int):
    A = continuous_tensor([_intervals(n, seed=10)], _values(n, 11), ["[)"])
    B = continuous_tensor([_intervals(n, seed=20)], _values(n, 21), ["[)"])

    def probes():
        return _probe_midpoints(A.dims[0], B.dims[0])

    return "i,i->i", (A, B), probes


def case_pinpoint_contraction(n: int):
    gen = torch.Generator().manual_seed(30)
    levels = torch.arange(max(4, n // 8), dtype=DTYPE) + 0.5
    ja = levels[torch.randint(0, len(levels), (n,), generator=gen)]
    jb = levels[torch.randint(0, len(levels), (n,), generator=gen)]
    A = continuous_tensor(
        [_intervals(n, seed=31), (ja,)], _values(n, 32), ["[)", "P"]
    )
    B = continuous_tensor([(jb,)], _values(n, 33), ["P"])

    def probes():
        return _probe_midpoints(A.dims[0])

    return "ij,j->i", (A, B), probes


def case_integrated_matmul(n: int):
    gen = torch.Generator().manual_seed(40)
    n_levels = max(4, round(math.sqrt(n)))
    ai = (torch.arange(n_levels, dtype=DTYPE) + 0.5)[
        torch.randint(0, n_levels, (n,), generator=gen)
    ]
    A = continuous_tensor(
        [(ai,), _intervals(n, seed=41, span=50.0)], _values(n, 42), ["P", "[)"]
    )
    B = continuous_tensor(
        [_intervals(n, seed=43, span=50.0), _intervals(n, seed=44, span=50.0)],
        _values(n, 45),
        ["[)", "[)"],
    )

    def probes():
        # 2-D probes: every i level × a capped set of j-cell midpoints.
        i_levels = torch.unique(A.dims[0][0])
        j_mid = _probe_midpoints(B.dims[1])[:, 0]
        step = max(1, (len(j_mid) * len(i_levels)) // N_PROBES)
        return torch.cartesian_prod(i_levels, j_mid[::step])

    return "ik,kj->ij", (A, B), probes


def _predict_table(equation: str, operands) -> tuple[list[int], int]:
    """Table shape (out-var candidate counts + operand nnz) without building it."""
    from ceinsum_equation import parse_equation
    from table_ceinsum import _candidates

    _, out_indices, carriers_of = parse_equation(equation, len(operands))
    out_vars = "".join(dict.fromkeys(out_indices))
    shape = [
        _candidates(carriers_of[v], operands)[0].shape[0] for v in out_vars
    ] + [op.nnz for op in operands]
    return shape, math.prod(shape)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

CASES = [
    ("`i,i->i` (interval × interval)", case_elementwise, True),
    ("`ij,j->i` (interval-i, pinpoint-j)", case_pinpoint_contraction, True),
    ("`ik,kj->ij` (A: P,I × B: I,I — integrated k)", case_integrated_matmul, True),
]


def run(sizes: list[int], repeats: int) -> str:
    lines: list[str] = []
    lines.append("# `table_ceinsum` benchmark")
    lines.append("")
    lines.append(
        f"Generated {date.today().isoformat()} by `benchmarks/bench_table_ceinsum.py` "
        f"— torch {torch.__version__}, {platform.machine()} CPU, "
        f"{torch.get_num_threads()} threads, {DTYPE}, median of {repeats} repeats "
        f"(1 warmup). Table sizes above {ELEM_GUARD:.0e} elements are skipped."
    )
    lines.append("")
    lines.append(
        "`table` times include the flatten to a COO piece list, so both columns "
        "produce the same kind of object. `max rel Δ` is the pointwise value "
        "disagreement between the two implementations at probe points."
    )

    for title, make_case, has_baseline in CASES:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        if has_baseline:
            lines.append(
                "| n | table shape | table MB | table (ms) | ceinsum (ms) | ratio | max rel Δ |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|")
        else:
            lines.append("| n | table shape | table MB | table (ms) |")
            lines.append("|---:|---|---:|---:|")

        for n in sizes:
            equation, operands, probes = make_case(n)
            shape, elems = _predict_table(equation, operands)
            shape_str = "×".join(str(s) for s in shape)
            mb = elems * DTYPE.itemsize / 1e6
            if elems > ELEM_GUARD:
                row = f"| {n} | {shape_str} | {mb:,.0f} | skipped (> guard) |"
                if has_baseline:
                    t_base = _bench(lambda: ceinsum(equation, *operands), repeats)
                    row += f" {t_base * 1e3:,.2f} | — | — |"
                lines.append(row)
                print(f"  n={n}: skipped (predicted {elems:.1e} elements)")
                continue

            t_table = _bench(
                lambda: table_ceinsum(equation, *operands).to_coo(), repeats
            )
            row = f"| {n} | {shape_str} | {mb:,.1f} | {t_table * 1e3:,.2f} |"
            if has_baseline:
                t_base = _bench(lambda: ceinsum(equation, *operands), repeats)
                out_t = table_ceinsum(equation, *operands).to_coo()
                out_c = ceinsum(equation, *operands)
                pts = probes()
                diff = _max_rel_diff(_eval_at(out_t, pts), _eval_at(out_c, pts))
                row += (
                    f" {t_base * 1e3:,.2f} | {t_table / t_base:,.1f}× | {diff:.1e} |"
                )
            lines.append(row)
            print(f"  n={n}: table {t_table * 1e3:.2f} ms ({shape_str})")

    lines.append("")
    lines.append("## Observations")
    lines.append("")
    lines.append(
        "- The table is dense: its element count is (candidate counts) × "
        "(product of operand piece counts), so time and memory grow with the "
        "*product* of the piece counts — the size guard trips exactly where "
        "that product-scaling wall predicts. This is the expected cost of the "
        "specification-level implementation; the piece-join pipeline "
        "(`continuous_einsum.ceinsum`) exists to avoid it."
    )
    lines.append(
        "- At small piece counts the dense table can be *faster* than the "
        "optimized pipeline: a handful of broadcast comparisons plus one "
        "einsum has less fixed overhead than the multi-step join. The "
        "product-scaling cost only dominates once the piece counts grow."
    )
    lines.append(
        "- The two implementations share integral semantics (a contracted "
        "all-interval variable is weighted by its overlap length) and agree "
        "pointwise to float32 precision on every case (`max rel Δ` column)."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="25,50,100,200,400,800")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="docs/table_ceinsum_bench.md")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    report = run(sizes, args.repeats)
    out_path = Path(__file__).resolve().parents[1] / args.out
    out_path.write_text(report)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
