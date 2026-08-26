"""Exp 1 — mask creation: dense vs binary-search vs polars db-join builders.

Times the three mask builders on six cases from the mapping suite
(``tests/test_mapping.py``), chosen to cover the distinct join condition
types: interval-interval overlap, point-in-interval containment, composite
equality, contraction geometry, mixed real-world joins, and a 3-operand case
whose dense table is N^3 (the OOM wall).

The dense and binary-search closures are wrapped in ``torch.compile`` (as in
the mapping bench). The db-join builder is timed on *execution only*: the
LazyFrame plan is built once outside the timed region and each repeat times
``plan.collect(engine=...)`` — GPU engine (cudf) in gpu device-mode, CPU
engine otherwise — mirroring the mapping bench's methodology (frame
construction and join_where AST building are input prep, analogous to the
torch.compile warmup the other builders get).

Usage:
    python benchmarks/exp1_mask_builders.py [--device-mode gpu|cpu-single|cpu-multi|all]
        [--sizes 100,300,...] [--skews 0.0,0.5,1.0] [--repeats 5] [--smoke]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import bench_common as bc

EXP1_CASE_LABELS = (
    "02_pointwise_1d_ii",   # interval-interval overlap
    "03_pointwise_1d_pi",   # point-in-interval containment
    "04_pointwise_2d_pp",   # composite 2-key equality
    "12_matmul_ii",         # contraction geometry
    "18_bio_intersect",     # mixed overlap + equality
    "16_triple_2d_ii",      # 3 operands: dense table is N^3
)


def _local_case_specs(test_mapping):
    """Extra mask cases defined here rather than in the test suite.

    ``20_pointwise_2d_ppii`` — A all-pinpoint 2D × B all-interval 2D
    (point-in-box on both axes): the missing property mix between the
    suite's all-pinpoint (04) and all-interval (05) pointwise cases.
    """
    from types import SimpleNamespace

    from mask_binary_search import build_binary_search_mask
    from mask_db_join import build_db_join_mask
    from mask_dense import build_dense_mask

    def build_ppii(n: int, skew: float):
        import polars as pl
        import torch

        a_x, a_y = test_mapping._pp_2d(n, skew, test_mapping.SEED_A)
        b_xs, b_xe, b_ys, b_ye = test_mapping._ii_2d(n, skew, test_mapping.SEED_B)
        op = {"A_x": a_x, "A_y": a_y, "B_x_s": b_xs, "B_x_e": b_xe,
              "B_y_s": b_ys, "B_y_e": b_ye}
        output = ("A", "B")
        eqs = [
            "B_x_s[B] <= A_x[A]", "A_x[A] < B_x_e[B]",
            "B_y_s[B] <= A_y[A]", "A_y[A] < B_y_e[B]",
        ]

        def polars_plan() -> "pl.LazyFrame":
            a_lf = pl.DataFrame({
                "A_piece": torch.arange(n, dtype=torch.long),
                "A_x": a_x.cpu().contiguous(),
                "A_y": a_y.cpu().contiguous(),
            }).lazy()
            b_lf = pl.DataFrame({
                "B_piece": torch.arange(n, dtype=torch.long),
                "B_x_s": b_xs.cpu().contiguous(),
                "B_x_e": b_xe.cpu().contiguous(),
                "B_y_s": b_ys.cpu().contiguous(),
                "B_y_e": b_ye.cpu().contiguous(),
            }).lazy()
            plan = a_lf.join_where(
                b_lf,
                pl.col("A_x") >= pl.col("B_x_s"),
                pl.col("A_x") < pl.col("B_x_e"),
                pl.col("A_y") >= pl.col("B_y_s"),
                pl.col("A_y") < pl.col("B_y_e"),
            )
            return plan.select(("A_piece", "B_piece"))

        return SimpleNamespace(
            table_auto=build_dense_mask(op, output, eqs),
            table_opt_auto=build_binary_search_mask(op, output, eqs),
            table_db_auto=build_db_join_mask(op, output, eqs),
            polars_plan=polars_plan,
            total_candidates=n * n,
        )

    return {
        "20_pointwise_2d_ppii": SimpleNamespace(
            label="20_pointwise_2d_ppii", build=build_ppii
        ),
    }

DEFAULT_SIZES = "100,300,1000,3000,10000,30000,100000"
DEFAULT_SKEWS = "0.0,0.5,1.0"

FIELDS = [
    "case", "skew", "n", "builder", "device_mode", "repeats", "warmup",
    "status", "alive_rows", "total_candidates",
    "time_ms_median", "time_ms_all", "note",
]

# Predicted dense boolean-table bytes above which the cell is skipped rather
# than attempted. Real allocation failures below the guard are still caught
# and recorded as ``oom``.
DENSE_GUARD_BYTES = {"cuda": 4e9, "cpu": 1.6e10}

# The polars join can exhaust *host* RAM, where the kernel OOM killer takes
# the whole process down before Python sees an exception (observed: the
# 3-operand case at n=100k, 1e15 candidates, killed at 128 GB RSS; 2.7e13
# candidates at n=30k ran fine). Skip db_join cells above this many
# candidates instead of attempting them.
DB_JOIN_GUARD_CANDIDATES = 1e14

# The builder-agreement cross-check runs all three builders at the smallest
# n of a (case, skew) sweep. Above this size that's itself an OOM risk (and
# agreement is already established by the main sweeps at small n), so skip.
AGREEMENT_N_MAX = 30000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device-mode", choices=(*bc.DEVICE_MODES, "all"), default="gpu")
    p.add_argument("--sizes", default=DEFAULT_SIZES)
    p.add_argument("--skews", default=DEFAULT_SKEWS)
    p.add_argument("--repeats", type=int, default=bc.DEFAULT_REPEATS)
    p.add_argument("--warmup", type=int, default=bc.DEFAULT_WARMUP)
    p.add_argument("--budget-s", type=float, default=bc.DEFAULT_BUDGET_S)
    p.add_argument("--out", default=str(bc.RESULTS_DIR / "exp1_mask_builders.csv"))
    p.add_argument("--smoke", action="store_true",
                   help="tiny sizes / 2 repeats, for validating the harness")
    p.add_argument("--resume", action="store_true",
                   help="skip cells already in the CSV; cells left in "
                        "'started' state are recorded as killed")
    p.add_argument("--cases", default=None,
                   help="comma-separated case labels to run (default: the "
                        "built-in six)")
    p.add_argument("--builders", default=None,
                   help="comma-separated builders to run "
                        "(dense,binary_search,db_join; default: all)")
    return p.parse_args(argv)


def passthrough_args(args) -> list[str]:
    out = ["--sizes", args.sizes, "--skews", args.skews,
           "--repeats", str(args.repeats), "--warmup", str(args.warmup),
           "--budget-s", str(args.budget_s), "--out", args.out]
    if args.smoke:
        out.append("--smoke")
    if args.resume:
        out.append("--resume")
    if args.cases:
        out.extend(["--cases", args.cases])
    if args.builders:
        out.extend(["--builders", args.builders])
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device_mode == "all":
        return bc.run_all_modes(__file__, passthrough_args(args))

    device = bc.apply_device_mode(args.device_mode)
    bc.add_src_to_path()
    bc.add_tests_to_path()

    import torch
    import test_mapping

    sizes = bc.parse_int_list(args.sizes)
    skews = bc.parse_float_list(args.skews)
    repeats = args.repeats
    if args.smoke:
        sizes, skews, repeats = [50, 100], [0.5], 2

    spec_by_label = {s.label: s for s in test_mapping.CASE_SPECS}
    spec_by_label.update(_local_case_specs(test_mapping))
    case_labels = EXP1_CASE_LABELS
    if args.cases:
        wanted = args.cases.split(",")
        unknown = [w for w in wanted if w not in spec_by_label]
        if unknown:
            raise SystemExit(f"unknown case labels: {unknown}")
        case_labels = tuple(wanted)
    key_cols = ("case", "skew", "n", "builder", "device_mode")
    prior = bc.load_last_status(Path(args.out), key_cols) if args.resume else {}
    writer = bc.CsvWriter(Path(args.out), FIELDS, device)
    ladder = bc.FailureLadder()
    guard_bytes = DENSE_GUARD_BYTES[device.type]

    def canonical(out) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return test_mapping._canonical_rows(out)
        return test_mapping._canonical_cols(out)

    def alive_of(out) -> int:
        if hasattr(out, "height"):  # polars DataFrame
            return int(out.height)
        if isinstance(out, torch.Tensor):
            return int(out.shape[0])
        return int(out[0].shape[0])

    polars_engine = "gpu" if device.type == "cuda" else "cpu"

    for label in case_labels:
        spec = spec_by_label[label]
        for skew in skews:
            agreement_done = False
            for n in sorted(sizes):
                case = spec.build(n, skew)  # data generation is untimed
                builders = (
                    ("dense", case.table_auto, True),
                    ("binary_search", case.table_opt_auto, True),
                    ("db_join", case.table_db_auto, False),
                )
                if args.builders:
                    allowed = set(args.builders.split(","))
                    builders = tuple(b for b in builders if b[0] in allowed)

                if not agreement_done and n > AGREEMENT_N_MAX:
                    agreement_done = True  # too large to verify safely here
                if not agreement_done:
                    # Cross-check all three builders once per (case, skew) at
                    # the smallest n; mismatches abort this (case, skew).
                    mismatch = None
                    try:
                        ref = canonical(case.table_opt_auto())
                        for name, closure, _ in builders:
                            if name == "binary_search":
                                continue
                            got = canonical(closure())
                            if got.shape != ref.shape or not torch.equal(got, ref):
                                mismatch = name
                                break
                    except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(f"{label} skew={skew:g}: agreement check skipped "
                              f"({type(e).__name__})", flush=True)
                    if mismatch is not None:
                        for name, _, _ in builders:
                            writer.write({
                                "case": label, "skew": skew, "n": n,
                                "builder": name, "device_mode": args.device_mode,
                                "repeats": repeats, "warmup": args.warmup,
                                "status": bc.STATUS_ERROR,
                                "alive_rows": "", "total_candidates": case.total_candidates,
                                "time_ms_median": "", "time_ms_all": "",
                                "note": f"builder disagreement: {mismatch} vs binary_search",
                            })
                        print(f"{label} skew={skew}: DISAGREEMENT ({mismatch}), skipping", flush=True)
                        break
                    agreement_done = True

                for name, closure, do_compile in builders:
                    key = (label, skew, name)
                    row = {
                        "case": label, "skew": skew, "n": n, "builder": name,
                        "device_mode": args.device_mode, "repeats": repeats,
                        "warmup": args.warmup, "alive_rows": "",
                        "total_candidates": case.total_candidates,
                        "time_ms_median": "", "time_ms_all": "", "note": "",
                    }
                    csv_key = (label, str(skew), str(n), name, args.device_mode)
                    if csv_key in prior:
                        last = prior[csv_key]
                        if last == bc.STATUS_STARTED:
                            row["status"] = bc.STATUS_KILLED
                            row["note"] = "left in 'started' state by a previous run"
                            writer.write(row)
                            ladder.record(key, n, bc.STATUS_KILLED)
                            print(f"{label} skew={skew:g} n={n} {name}: killed (resume)", flush=True)
                            continue
                        if last != bc.STATUS_PRIOR:
                            ladder.record(key, n, last)  # seed the skip ladder
                            continue
                        # skipped_prior_fail is not a measurement — fall
                        # through so a targeted rerun can attempt the cell.
                    if ladder.should_skip(key, n):
                        row["status"] = bc.STATUS_PRIOR
                        writer.write(row)
                        continue
                    if name == "dense" and case.total_candidates > guard_bytes:
                        row["status"] = bc.STATUS_GUARD
                        row["note"] = f"predicted {case.total_candidates:.1e} B > {guard_bytes:.1e} B"
                        ladder.record(key, n, bc.STATUS_BUDGET)  # skip larger n too
                        writer.write(row)
                        print(f"{label} skew={skew:g} n={n} {name}: guard", flush=True)
                        continue
                    if name == "db_join" and case.total_candidates > DB_JOIN_GUARD_CANDIDATES:
                        row["status"] = bc.STATUS_GUARD
                        row["note"] = (f"{case.total_candidates:.1e} candidates > "
                                       f"{DB_JOIN_GUARD_CANDIDATES:.0e} (host-RAM kill risk)")
                        ladder.record(key, n, bc.STATUS_BUDGET)
                        writer.write(row)
                        print(f"{label} skew={skew:g} n={n} {name}: guard", flush=True)
                        continue

                    writer.write(dict(row, status=bc.STATUS_STARTED))
                    if name == "db_join":
                        # Execution only: the plan is input prep (built once,
                        # untimed), each repeat times the collect on the
                        # engine matching the device mode.
                        plan = case.polars_plan()
                        fn = lambda: plan.collect(engine=polars_engine)  # noqa: E731
                        row["note"] = f"engine={polars_engine}"
                    else:
                        fn = test_mapping._compile_table_fn(closure) if do_compile else closure
                    alive: list[int] = []
                    res = bc.timed_cell(
                        fn, repeats=repeats, device=device, warmup=args.warmup,
                        budget_s=args.budget_s,
                        probe=lambda out: alive.append(alive_of(out)),
                    )
                    ladder.record(key, n, res["status"])
                    row.update({
                        "status": res["status"],
                        "alive_rows": alive[0] if alive else "",
                        "time_ms_median": bc.fmt_ms(res["median_ms"]),
                        "time_ms_all": bc.fmt_all(res["all_ms"]),
                        "note": "; ".join(s for s in (row["note"], res["note"]) if s),
                    })
                    writer.write(row)
                    print(
                        f"{label} skew={skew:g} n={n} {name}: "
                        f"{res['status']} {bc.fmt_ms(res['median_ms'])} ms",
                        flush=True,
                    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
