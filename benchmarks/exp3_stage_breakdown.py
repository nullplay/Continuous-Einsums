"""Exp 3 — stage breakdown: mask / product / merge timing.

Runs the pipeline stages directly (no core changes: they are the same pure
functions ``ceinsum`` orchestrates) with CUDA synchronization at every stage
boundary, so each stage is timed on the exact intermediates the previous
stage produced. Three representative cases:

* ``matmul_ii``      (skew 0.5) — mask-dominated contraction.
* ``matvec_ii``      (skew 1.0) — merge-dominated: clustering maximizes
                                  overlapping candidates on the output axis.
* ``pointwise_2d_ii`` (skew 0.5) — no-reduction control: merge is the
                                  assembly branch, near-zero by design.

``total_e2e`` rows time the full ``ceinsum()`` call with the same repeats;
the stage sum is checked against it (warning above 10 %).

Usage:
    python benchmarks/exp3_stage_breakdown.py [--device-mode gpu|cpu-single|cpu-multi|all]
        [--sizes 1000,10000,100000] [--repeats 7] [--smoke]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import bench_common as bc

EXP3_CASES = (  # (label, skew, sizes override or None)
    # matmul's merged output explodes (~25M pieces at n=3000, 9.7 GB peak on
    # the RTX 3090); n=10000 OOMs the GPU, so it sweeps smaller sizes.
    ("matmul_ii", 0.5, (300, 1000, 3000)),
    ("matvec_ii", 1.0, None),
    ("pointwise_2d_ii", 0.5, None),
)

DEFAULT_SIZES = "1000,10000,100000"
DEFAULT_REPEATS = 7

FIELDS = [
    "device_mode", "case", "equation", "n", "skew", "stage",
    "repeats", "status", "rows", "time_ms_median", "time_ms_all", "note",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device-mode", choices=(*bc.DEVICE_MODES, "all"), default="gpu")
    p.add_argument("--sizes", default=DEFAULT_SIZES)
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--warmup", type=int, default=bc.DEFAULT_WARMUP)
    p.add_argument("--budget-s", type=float, default=bc.DEFAULT_BUDGET_S)
    p.add_argument("--out", default=str(bc.RESULTS_DIR / "exp3_stages.csv"))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def passthrough_args(args) -> list[str]:
    out = ["--sizes", args.sizes, "--repeats", str(args.repeats),
           "--warmup", str(args.warmup), "--budget-s", str(args.budget_s),
           "--out", args.out]
    if args.smoke:
        out.append("--smoke")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device_mode == "all":
        return bc.run_all_modes(__file__, passthrough_args(args))

    device = bc.apply_device_mode(args.device_mode)
    bc.add_src_to_path()

    import torch
    from ceinsum_equation import parse_equation
    from ceinsum_mask import build_mask
    from ceinsum_merge import merge
    from ceinsum_product import compute_output_properties, compute_product
    from continuous_einsum import ceinsum
    from ctensor import ContinuousTensor

    from ceinsum_cases import CASE_BY_LABEL

    sizes = sorted(bc.parse_int_list(args.sizes))
    repeats = args.repeats
    if args.smoke:
        sizes, repeats = [100], 2

    writer = bc.CsvWriter(Path(args.out), FIELDS, device)
    cuda = device.type == "cuda"

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    for label, skew, size_override in EXP3_CASES:
        case = CASE_BY_LABEL[label]
        case_sizes = sizes if size_override is None or args.smoke else sorted(size_override)
        for n in case_sizes:
            operands = case.make_operands(n, skew, device)

            # Untimed setup: parse + output properties (µs-level, outside the
            # mask/product/merge framing, same split as ceinsum itself).
            in_indices, out_indices, i2od = parse_equation(
                case.equation, len(operands)
            )
            index_props = compute_output_properties(operands, i2od, out_indices)
            out_property = tuple(index_props[oi] for oi in out_indices)

            def one_pass() -> tuple[float, float, float, int, int]:
                """One full pipeline pass; returns per-stage seconds + sizes."""
                sync()
                t0 = time.perf_counter()
                mask = build_mask(operands, i2od, out_indices, None)
                sync()
                t1 = time.perf_counter()
                product = compute_product(
                    operands, mask, i2od, out_indices, index_props
                )
                sync()
                t2 = time.perf_counter()
                if case.has_reduction:
                    out = merge(product.coords, product.values, out_property)
                else:
                    out = ContinuousTensor(
                        tuple(tuple(spec) for spec in product.coords),
                        product.values, out_property,
                    )
                sync()
                t3 = time.perf_counter()
                mask_entries = int(mask.piece_idx[0].shape[0])
                out_nnz = out.nnz
                del mask, product, out
                if cuda:
                    torch.cuda.empty_cache()
                return t1 - t0, t2 - t1, t3 - t2, mask_entries, out_nnz

            base_row = {
                "device_mode": args.device_mode, "case": label,
                "equation": case.equation, "n": n, "skew": skew,
                "repeats": repeats, "note": "",
            }
            try:
                first = one_pass()
                if sum(first[:3]) > args.budget_s:
                    for stage, t, rows in zip(
                        ("mask", "product", "merge"), first[:3],
                        (first[3], first[3], first[4]),
                    ):
                        writer.write(dict(
                            base_row, stage=stage, status=bc.STATUS_BUDGET,
                            rows=rows, time_ms_median=bc.fmt_ms(t * 1e3),
                            time_ms_all=bc.fmt_all([t * 1e3]),
                            note=f"first pass {sum(first[:3]):.1f}s > budget",
                        ))
                    print(f"{label} n={n}: budget", flush=True)
                    continue
                for _ in range(args.warmup - 1):
                    one_pass()
                passes = [one_pass() for _ in range(repeats)]
            except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                if cuda:
                    torch.cuda.empty_cache()
                for stage in ("mask", "product", "merge"):
                    writer.write(dict(
                        base_row, stage=stage, status=bc.STATUS_OOM, rows="",
                        time_ms_median="", time_ms_all="",
                        note=type(e).__name__,
                    ))
                print(f"{label} n={n}: OOM", flush=True)
                continue

            mask_entries, out_nnz = passes[0][3], passes[0][4]
            stage_medians: dict[str, float] = {}
            for idx, (stage, rows) in enumerate(
                (("mask", mask_entries), ("product", mask_entries),
                 ("merge", out_nnz))
            ):
                all_ms = [p[idx] * 1e3 for p in passes]
                med = statistics.median(all_ms)
                stage_medians[stage] = med
                writer.write(dict(
                    base_row, stage=stage, status=bc.STATUS_OK, rows=rows,
                    time_ms_median=bc.fmt_ms(med),
                    time_ms_all=bc.fmt_all(all_ms),
                ))

            # End-to-end reference with the same repeats.
            res = bc.timed_cell(
                lambda: ceinsum(case.equation, *operands),
                repeats=repeats, device=device, warmup=args.warmup,
                budget_s=args.budget_s,
            )
            note = res["note"]
            if res["status"] == bc.STATUS_OK:
                stage_sum = sum(stage_medians.values())
                gap = abs(stage_sum - res["median_ms"]) / max(res["median_ms"], 1e-9)
                if gap > 0.10:
                    note = f"stage sum {stage_sum:.3f} ms vs e2e gap {gap:.0%}"
                    print(f"  WARNING {label} n={n}: {note}", flush=True)
            writer.write(dict(
                base_row, stage="total_e2e", status=res["status"], rows=out_nnz,
                time_ms_median=bc.fmt_ms(res["median_ms"]),
                time_ms_all=bc.fmt_all(res["all_ms"]), note=note,
            ))
            print(
                f"{label} n={n}: mask {stage_medians['mask']:.3f} / "
                f"product {stage_medians['product']:.3f} / "
                f"merge {stage_medians['merge']:.3f} / "
                f"e2e {bc.fmt_ms(res['median_ms'])} ms",
                flush=True,
            )

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
