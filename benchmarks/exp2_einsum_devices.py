"""Exp 2 — end-to-end ``ceinsum`` across device modes.

Times the full mask → product → merge pipeline (default binary-search mask
builder) on the 7 equation-level cases in ``ceinsum_cases.py``, one device
mode per process (cpu-single / cpu-multi / gpu). The dense-table reference
``table_ceinsum`` is additionally timed at small sizes, guarded by its
predicted element count — the guard trips are the dense memory wall, reported
as ``skipped_guard`` rather than failures.

At n ≤ 400 the ceinsum output is cross-checked pointwise against
``table_ceinsum`` at probe points (the two piece lists differ by
construction, so values are compared, not pieces).

Usage:
    python benchmarks/exp2_einsum_devices.py [--device-mode gpu|cpu-single|cpu-multi|all]
        [--sizes 100,300,...] [--skew 0.5] [--repeats 5] [--smoke]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bench_common as bc

DEFAULT_SIZES = "100,300,1000,3000,10000,30000,100000"
BASELINE_SIZES = (25, 50, 100, 200, 400, 800)
CHECK_N_MAX = 400
N_PROBES = 512
DIFF_FLAG = 1e-4

FIELDS = [
    "device_mode", "case", "equation", "n", "skew", "impl",
    "repeats", "warmup", "status", "out_nnz", "max_rel_diff",
    "time_ms_median", "time_ms_all", "note",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device-mode", choices=(*bc.DEVICE_MODES, "all"), default="gpu")
    p.add_argument("--sizes", default=DEFAULT_SIZES)
    p.add_argument("--skew", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=bc.DEFAULT_REPEATS)
    p.add_argument("--warmup", type=int, default=bc.DEFAULT_WARMUP)
    p.add_argument("--budget-s", type=float, default=bc.DEFAULT_BUDGET_S)
    p.add_argument("--out", default=str(bc.RESULTS_DIR / "exp2_einsum.csv"))
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the table_ceinsum baseline rows")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="skip cells already in the CSV; cells left in "
                        "'started' state are recorded as killed")
    p.add_argument("--cases", default=None,
                   help="comma-separated case labels to run (default: all)")
    return p.parse_args(argv)


def passthrough_args(args) -> list[str]:
    out = ["--sizes", args.sizes, "--skew", str(args.skew),
           "--repeats", str(args.repeats), "--warmup", str(args.warmup),
           "--budget-s", str(args.budget_s), "--out", args.out]
    if args.no_baseline:
        out.append("--no-baseline")
    if args.smoke:
        out.append("--smoke")
    if args.resume:
        out.append("--resume")
    if args.cases:
        out.extend(["--cases", args.cases])
    return out


def _generic_probes(ct, n_probes: int = N_PROBES):
    """Probe points derived from the output's own piece geometry.

    Per interval dim: midpoints between consecutive distinct endpoints;
    per pinpoint dim: the coordinates themselves. Capped near ``n_probes``
    via an even per-dim subsample before the cartesian product.
    """
    import torch

    if ct.ndim == 0:
        return None
    per_dim = []
    k = max(2, int(math.ceil(n_probes ** (1.0 / ct.ndim))))
    for d in range(ct.ndim):
        spec = ct.dims[d]
        pts = torch.unique(torch.cat([t.detach().cpu() for t in spec]))
        cand = pts if len(spec) == 1 else (pts[:-1] + pts[1:]) / 2
        if cand.numel() == 0:
            cand = torch.zeros(1)
        if cand.numel() > k:
            step = cand.numel() // k
            cand = cand[::step][:k]
        per_dim.append(cand)
    if ct.ndim == 1:
        return per_dim[0][:, None]
    return torch.cartesian_prod(*per_dim)


def _eval_at(ct, points):
    """Sum of piece values covering each probe point (CPU, float64)."""
    import torch

    if ct.ndim == 0:
        return ct.values.detach().cpu().to(torch.float64).sum().reshape(1)
    npts = points.shape[0]
    vals = ct.values.detach().cpu().to(torch.float64)
    mask = torch.ones(npts, ct.nnz, dtype=torch.bool)
    for d in range(ct.ndim):
        p = points[:, d][:, None]
        spec = tuple(t.detach().cpu() for t in ct.dims[d])
        if len(spec) == 1:
            mask &= p == spec[0][None, :]
        else:
            mask &= (spec[0][None, :] <= p) & (p < spec[1][None, :])
    return (mask.to(torch.float64) * vals[None, :]).sum(dim=1)


def _max_rel_diff(a, b) -> float:
    import torch

    scale = torch.maximum(a.abs(), b.abs()).clamp(min=1.0)
    return float(((a - b).abs() / scale).max()) if a.numel() else 0.0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device_mode == "all":
        return bc.run_all_modes(__file__, passthrough_args(args))

    device = bc.apply_device_mode(args.device_mode)
    bc.add_src_to_path()

    import torch
    from continuous_einsum import ceinsum
    from table_ceinsum import table_ceinsum
    from bench_table_ceinsum import ELEM_GUARD, _predict_table

    from ceinsum_cases import CEINSUM_CASES

    sizes = sorted(bc.parse_int_list(args.sizes))
    baseline_sizes = list(BASELINE_SIZES)
    repeats = args.repeats
    if args.smoke:
        sizes, baseline_sizes, repeats = [50, 100], [50], 2

    key_cols = ("device_mode", "case", "n", "skew", "impl")
    prior = bc.load_last_status(Path(args.out), key_cols) if args.resume else {}
    writer = bc.CsvWriter(Path(args.out), FIELDS, device)
    ladder = bc.FailureLadder()

    run_cases = CEINSUM_CASES
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - {c.label for c in CEINSUM_CASES}
        if unknown:
            raise SystemExit(f"unknown case labels: {sorted(unknown)}")
        run_cases = tuple(c for c in CEINSUM_CASES if c.label in wanted)

    def resume_hit(row: dict, key, n: int) -> bool:
        """True when --resume already covers this cell (writes killed rows)."""
        csv_key = (args.device_mode, row["case"], str(n), str(args.skew), row["impl"])
        if csv_key not in prior:
            return False
        last = prior[csv_key]
        if last == bc.STATUS_STARTED:
            writer.write(dict(row, status=bc.STATUS_KILLED,
                              note="left in 'started' state by a previous run"))
            ladder.record(key, n, bc.STATUS_KILLED)
            print(f"{row['case']} n={n} {row['impl']}: killed (resume)", flush=True)
            return True
        if last == bc.STATUS_PRIOR:
            # Not a measurement — let a targeted rerun attempt the cell.
            return False
        ladder.record(key, n, last)
        return True

    for case in run_cases:
        base_row = {
            "device_mode": args.device_mode, "case": case.label,
            "equation": case.equation, "skew": args.skew,
            "repeats": repeats, "warmup": args.warmup,
        }

        # ---- ceinsum pipeline ----
        for n in sizes:
            key = (case.label, "ceinsum")
            row = dict(base_row, n=n, impl="ceinsum", out_nnz="",
                       max_rel_diff="", time_ms_median="", time_ms_all="", note="")
            if resume_hit(row, key, n):
                continue
            if ladder.should_skip(key, n):
                row["status"] = bc.STATUS_PRIOR
                writer.write(row)
                continue

            writer.write(dict(row, status=bc.STATUS_STARTED))
            operands = case.make_operands(n, args.skew, device)
            nnz: list[int] = []
            res = bc.timed_cell(
                lambda: ceinsum(case.equation, *operands),
                repeats=repeats, device=device, warmup=args.warmup,
                budget_s=args.budget_s,
                probe=lambda out: nnz.append(out.nnz),
            )
            ladder.record(key, n, res["status"])

            diff = ""
            if res["status"] == bc.STATUS_OK and n <= CHECK_N_MAX:
                # Cross-check pointwise against the dense-table reference.
                try:
                    out_c = ceinsum(case.equation, *operands)
                    _, elems = _predict_table(case.equation, operands)
                    if elems <= ELEM_GUARD:
                        out_ref = table_ceinsum(case.equation, *operands).to_coo()
                        ref_name = "table_ceinsum"
                    else:
                        from mask_dense import build_dense_mask

                        out_ref = ceinsum(case.equation, *operands,
                                          builder=build_dense_mask)
                        ref_name = "ceinsum+dense"
                    pts = _generic_probes(out_c)
                    d = _max_rel_diff(_eval_at(out_c, pts), _eval_at(out_ref, pts))
                    diff = f"{d:.2e}"
                    if d > DIFF_FLAG:
                        row["note"] = f"max_rel_diff {d:.2e} > {DIFF_FLAG} vs {ref_name}"
                except Exception as e:  # noqa: BLE001
                    row["note"] = f"crosscheck failed: {type(e).__name__}: {e}"

            row.update({
                "status": res["status"],
                "out_nnz": nnz[0] if nnz else "",
                "max_rel_diff": diff,
                "time_ms_median": bc.fmt_ms(res["median_ms"]),
                "time_ms_all": bc.fmt_all(res["all_ms"]),
                "note": row["note"] or res["note"],
            })
            writer.write(row)
            print(f"{case.label} n={n} ceinsum: {res['status']} "
                  f"{bc.fmt_ms(res['median_ms'])} ms", flush=True)

        # ---- table_ceinsum baseline ----
        if args.no_baseline:
            continue
        for n in baseline_sizes:
            key = (case.label, "table")
            row = dict(base_row, n=n, impl="table_ceinsum", out_nnz="",
                       max_rel_diff="", time_ms_median="", time_ms_all="", note="")
            if resume_hit(row, key, n):
                continue
            if ladder.should_skip(key, n):
                row["status"] = bc.STATUS_PRIOR
                writer.write(row)
                continue

            operands = case.make_operands(n, args.skew, device)
            shape, elems = _predict_table(case.equation, operands)
            if elems > ELEM_GUARD:
                row["status"] = bc.STATUS_GUARD
                row["note"] = (f"predicted table {elems:.1e} elems "
                               f"({'×'.join(map(str, shape))}) > {ELEM_GUARD:.0e}")
                ladder.record(key, n, bc.STATUS_BUDGET)
                writer.write(row)
                print(f"{case.label} n={n} table: guard ({elems:.1e})", flush=True)
                continue

            writer.write(dict(row, status=bc.STATUS_STARTED))
            nnz = []
            res = bc.timed_cell(
                lambda: table_ceinsum(case.equation, *operands).to_coo(),
                repeats=repeats, device=device, warmup=args.warmup,
                budget_s=args.budget_s,
                probe=lambda out: nnz.append(out.nnz),
            )
            ladder.record(key, n, res["status"])
            row.update({
                "status": res["status"],
                "out_nnz": nnz[0] if nnz else "",
                "time_ms_median": bc.fmt_ms(res["median_ms"]),
                "time_ms_all": bc.fmt_all(res["all_ms"]),
                "note": res["note"],
            })
            writer.write(row)
            print(f"{case.label} n={n} table: {res['status']} "
                  f"{bc.fmt_ms(res['median_ms'])} ms", flush=True)

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
