"""Regenerate thesis figures (PDF) from the benchmark CSVs — no reruns.

Reads ``benchmarks/results/*.csv`` and writes ``benchmarks/figures/*.pdf``:

* Exp 1: ``exp1_<case>.pdf`` — one subplot per skew, log-log n vs median ms,
  one line per mask builder; OOM/guard/budget walls annotated where a
  builder's line ends.
* Exp 2: ``exp2_<case>.pdf`` — log-log n vs median ms, one line per device
  mode, ``table_ceinsum`` baseline dashed in the same hue, guard wall
  annotated.
* Exp 3: ``exp3_stages.pdf`` — per case: grouped bars (log y) of the
  mask/product/merge medians, plus a 100%-stacked share panel (stacked bars
  on a log axis would misrepresent the segment sums, hence the split).

If a CSV holds several rows for the same measurement key (appended runs),
the last row wins.

Usage:  python benchmarks/plot_results.py [--exp 1|2|3|all]
        [--results-dir benchmarks/results] [--out-dir benchmarks/figures]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bench_common as bc

# Validated categorical palette (dataviz reference, slots 1-3; CVD ΔE 47 on
# white). Aqua/yellow sit below 3:1 on white — relieved by distinct markers,
# a legend, and end-of-line annotations.
SERIES_COLORS = ("#2a78d6", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^")

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed slot assignment — color follows the entity across every figure.
BUILDER_ORDER = ("dense", "binary_search", "db_join")
MODE_ORDER = ("cpu-single", "cpu-multi", "gpu")
STAGE_ORDER = ("mask", "product", "merge")

FAIL_LABEL = {
    bc.STATUS_OOM: "OOM",
    # Guard trips are predicted OOMs (skipped before allocating) — label
    # them as what they are to the reader.
    bc.STATUS_GUARD: "OOM",
    bc.STATUS_BUDGET: ">budget",
    # Errors in this dataset are capacity failures (e.g. cudf's inequality
    # join overflowing its 32-bit internals) — read as OOM.
    bc.STATUS_ERROR: "OOM",
    bc.STATUS_KILLED: "OOM",  # host-RAM kill — an OOM to the reader
    bc.STATUS_PRIOR: None,  # implied by the earlier failure
    bc.STATUS_STARTED: None,  # sentinel, superseded or killed
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def dedupe(rows: list[dict], key_cols: tuple[str, ...]) -> list[dict]:
    """Last row wins per key (re-runs append to the CSV)."""
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key[tuple(r[c] for c in key_cols)] = r
    return list(by_key.values())


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.set_axisbelow(True)


def plot_series_with_walls(ax, rows, color, marker, label):
    """Log-log line of ok/budget cells + annotation where the series dies.

    ``rows`` are this series' rows sorted by n. Budget cells (one timed run)
    plot with an open marker; the first hard failure annotates the wall.
    """
    xs, ys, open_marker = [], [], []
    wall = None
    for r in rows:
        status = r["status"]
        if status in (bc.STATUS_OK, bc.STATUS_BUDGET) and r["time_ms_median"]:
            xs.append(int(r["n"]))
            ys.append(float(r["time_ms_median"]))
            open_marker.append(status == bc.STATUS_BUDGET)
        elif FAIL_LABEL.get(status) and wall is None:
            wall = (int(r["n"]), FAIL_LABEL[status])
    if xs:
        ax.plot(xs, ys, color=color, marker=marker, markersize=5,
                linewidth=1.5, label=label, zorder=3)
        for x, y, is_open in zip(xs, ys, open_marker):
            if is_open:
                ax.plot([x], [y], marker=marker, markersize=5, zorder=4,
                        markerfacecolor="white", markeredgecolor=color, ls="none")
    if wall and ys:
        ax.plot([wall[0]], [ys[-1]], marker="x", color=color, markersize=7,
                markeredgewidth=1.8, ls="none", zorder=4, clip_on=False)
        ax.annotate(wall[1], (wall[0], ys[-1]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color=color)
    return bool(xs)


def plot_exp1(results_dir: Path, out_dir: Path) -> None:
    rows = dedupe(
        load_csv(results_dir / "exp1_mask_builders.csv"),
        ("case", "skew", "n", "builder", "device_mode"),
    )
    if not rows:
        print("exp1: no data")
        return
    cases = sorted({r["case"] for r in rows})
    for case in cases:
        crows = [r for r in rows if r["case"] == case]
        skews = sorted({float(r["skew"]) for r in crows})
        fig, axes = plt.subplots(
            1, len(skews), figsize=(3.1 * len(skews), 3.0),
            sharey=True, squeeze=False,
        )
        for ax, skew in zip(axes[0], skews):
            srows = [r for r in crows if float(r["skew"]) == skew]
            for i, builder in enumerate(BUILDER_ORDER):
                brows = sorted(
                    (r for r in srows if r["builder"] == builder),
                    key=lambda r: int(r["n"]),
                )
                plot_series_with_walls(
                    ax, brows, SERIES_COLORS[i], MARKERS[i], builder
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"skew {skew:g}")
            ax.set_xlabel("pieces per operand (n)")
            style_axes(ax)
        axes[0][0].set_ylabel("mask build time (ms, median)")
        axes[0][0].legend(loc="upper left")
        fig.suptitle(f"Mask builders — {case}", fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = out_dir / f"exp1_{case}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


def plot_exp2(results_dir: Path, out_dir: Path) -> None:
    rows = dedupe(
        load_csv(results_dir / "exp2_einsum.csv"),
        ("device_mode", "case", "n", "skew", "impl"),
    )
    if not rows:
        print("exp2: no data")
        return
    cases = sorted({r["case"] for r in rows})
    for case in cases:
        crows = [r for r in rows if r["case"] == case]
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        for i, mode in enumerate(MODE_ORDER):
            main = sorted(
                (r for r in crows
                 if r["device_mode"] == mode and r["impl"] == "ceinsum"),
                key=lambda r: int(r["n"]),
            )
            plot_series_with_walls(ax, main, SERIES_COLORS[i], MARKERS[i], mode)
            base = sorted(
                (r for r in crows
                 if r["device_mode"] == mode and r["impl"] == "table_ceinsum"),
                key=lambda r: int(r["n"]),
            )
            bx = [int(r["n"]) for r in base
                  if r["status"] == bc.STATUS_OK and r["time_ms_median"]]
            by = [float(r["time_ms_median"]) for r in base
                  if r["status"] == bc.STATUS_OK and r["time_ms_median"]]
            if bx:
                ax.plot(bx, by, color=SERIES_COLORS[i], linewidth=1.2,
                        linestyle="--", alpha=0.85, zorder=2)
                walled = [r for r in base if r["status"] == bc.STATUS_GUARD]
                if walled:
                    ax.plot([bx[-1]], [by[-1]], marker="x",
                            color=SERIES_COLORS[i], markersize=6,
                            markeredgewidth=1.5, ls="none", zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("pieces per operand (n)")
        ax.set_ylabel("end-to-end time (ms, median)")
        style_axes(ax)
        eq = crows[0]["equation"]
        ax.set_title(f"{case}  ({eq})")
        handles = [
            Line2D([], [], color=SERIES_COLORS[i], marker=MARKERS[i],
                   markersize=5, linewidth=1.5, label=mode)
            for i, mode in enumerate(MODE_ORDER)
        ]
        handles.append(Line2D([], [], color=INK_2, linewidth=1.2,
                              linestyle="--", label="table ref (×: OOM)"))
        ax.legend(handles=handles, loc="upper left")
        fig.tight_layout()
        out = out_dir / f"exp2_{case}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


# Exp1 summary: three property-representative mask cases — only pinpoints,
# only intervals, mixed — labeled with their einsum-equivalent notation.
EXP1_SUMMARY_CASES = (
    "04_pointwise_2d_pp",
    "05_pointwise_2d_ii",
    "20_pointwise_2d_ppii",
)
EXP1_CASE_MATH = {
    "04_pointwise_2d_pp": r"$C^{PP}_{xy} = A^{PP}_{xy}\,B^{PP}_{xy}$",
    "05_pointwise_2d_ii": r"$C^{II}_{xy} = A^{II}_{xy}\,B^{II}_{xy}$",
    "20_pointwise_2d_ppii": r"$C^{PP}_{xy} = A^{PP}_{xy}\,B^{II}_{xy}$",
}


def plot_exp1_summary(results_dir: Path, out_dir: Path,
                      summary_n: int | None = None) -> None:
    """One row of three n-sweep subplots (pinpoints / intervals / mixed),
    log-log, one line per mask builder, at skew 0.5. Failure walls
    (guard/OOM/budget) are annotated where a builder's line ends."""
    del summary_n  # sweep figure — the whole n range is always shown
    rows = dedupe(
        load_csv(results_dir / "exp1_mask_builders.csv"),
        ("case", "skew", "n", "builder", "device_mode"),
    )
    rows = [r for r in rows
            if r["case"] in EXP1_SUMMARY_CASES and float(r["skew"]) == 0.5]
    if not rows:
        print("exp1 summary: no data")
        return

    fig, axes = plt.subplots(
        1, len(EXP1_SUMMARY_CASES), figsize=(3.1 * len(EXP1_SUMMARY_CASES), 3.2),
        sharey=True, squeeze=False,
    )
    for ax, case in zip(axes[0], EXP1_SUMMARY_CASES):
        crows = [r for r in rows if r["case"] == case]
        for i, builder in enumerate(BUILDER_ORDER):
            brows = sorted(
                (r for r in crows if r["builder"] == builder),
                key=lambda r: int(r["n"]),
            )
            plot_series_with_walls(ax, brows, SERIES_COLORS[i], MARKERS[i],
                                   builder)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(EXP1_CASE_MATH[case], fontsize=10)
        ax.set_xlabel("pieces per operand (n)")
        style_axes(ax)
    axes[0][0].set_ylabel("mask build time (ms, median)")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    out = out_dir / "exp1_summary.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# Math-notation x labels for the exp2 summary: indices x/y (r for the
# reduced index), operand properties as I/P superscripts. The triple case
# names its output D because C is its third operand.
CASE_MATH = {
    "pointwise_1d_ii": r"$C^{I}_{x} = A^{I}_{x}\,B^{I}_{x}$",
    "pointwise_2d_ii": r"$C^{II}_{xy} = A^{II}_{xy}\,B^{II}_{xy}$",
    "diagonal_ii_i": r"$C^{I}_{x} = A^{II}_{xx}\,B^{I}_{x}$",
    "matvec_ii": r"$C^{I}_{x} = A^{II}_{xr}\,B^{I}_{r}$",
    "matmul_ii": r"$C^{II}_{xy} = A^{II}_{xr}\,B^{II}_{ry}$",
    "triple_contract_pii": r"$D^{PP}_{xy} = A^{PP}_{xy}\,B^{II}_{xr}\,C^{II}_{ry}$",
    "box_search": r"$C^{PP}_{xy} = A^{II}_{xy}\,B^{PP}_{xy}$",
}
CASE_ORDER = tuple(CASE_MATH)


def plot_exp2_summary(results_dir: Path, out_dir: Path,
                      summary_n: int | None = None) -> None:
    """One grouped bar chart: every einsum × device mode at one size.

    Default size is the largest n at which every case has an ok ceinsum cell
    in all three device modes (no bar missing). An explicit ``summary_n``
    may exceed that; cells without data are annotated with the reason the
    series died (OOM / budget) instead of a bar.
    """
    rows = dedupe(
        load_csv(results_dir / "exp2_einsum.csv")
        + load_csv(results_dir / "exp2_finch.csv"),
        ("device_mode", "case", "n", "skew", "impl"),
    )
    rows = [r for r in rows if r["impl"] == "ceinsum"]
    if not rows:
        print("exp2 summary: no data")
        return
    # Finch rows (single-thread Julia, results/exp2_finch.csv) join as the
    # leftmost series; the default-n coverage rule stays over MODE_ORDER so a
    # partial Finch sweep can't shrink the chosen n. The finch bar shows
    # conversion + execution stacked: its COO->Finch level conversion
    # (convert_ms in the CSV note) is work the torch pipeline doesn't have.
    have_finch = any(r["device_mode"] == "finch" for r in rows)
    modes = (("finch",) if have_finch else ()) + tuple(MODE_ORDER)
    colors = (("#c2483b",) if have_finch else ()) + SERIES_COLORS
    med = {
        (r["case"], int(r["n"]), r["device_mode"]): float(r["time_ms_median"])
        for r in rows if r["status"] == bc.STATUS_OK and r["time_ms_median"]
        # Finch cannot express the ii diagonal (A[i,i] does not lower); its
        # numbers time a pre-extracted 1D kernel, so the cell shows n/a
        # rather than an incomparable bar.
        and not (r["device_mode"] == "finch" and r["case"] == "diagonal_ii_i")
    }
    convert = {
        (r["case"], int(r["n"])): float(m.group(1))
        for r in rows if r["device_mode"] == "finch"
        for m in [re.search(r"convert_ms=([\d.]+)", r["note"] or "")] if m
    }
    if summary_n is None:
        full_ns = [
            n for n in sorted({int(r["n"]) for r in rows})
            if all((c, n, m) in med for c in CASE_ORDER for m in MODE_ORDER)
        ]
        if not full_ns:
            print("exp2 summary: no n with full case × mode coverage")
            return
        n = full_ns[-1]
        suffix = ""
    else:
        n = summary_n
        suffix = f"_n{n}"
    skew = rows[0]["skew"]

    def fail_reason(case: str, mode: str) -> str:
        """Why this (case, mode) has no bar: its first hard failure."""
        for r in sorted((r for r in rows
                         if r["case"] == case and r["device_mode"] == mode
                         and int(r["n"]) <= n),
                        key=lambda r: int(r["n"])):
            if r["status"] in (bc.STATUS_OOM, bc.STATUS_KILLED):
                return "OOM"
            if r["status"] == bc.STATUS_BUDGET:
                return ">2 min"
        return "n/a"

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    width = 0.78 / len(modes)
    x = range(len(CASE_ORDER))
    floor = min(med[k] for k in med) if med else 1.0
    missing: dict[int, list[tuple[float, str, int]]] = {}  # case idx -> cells
    for i, mode in enumerate(modes):
        xs, ys, cs = [], [], []
        for xi, c in zip(x, CASE_ORDER):
            xpos = xi + (i - (len(modes) - 1) / 2) * width
            v = med.get((c, n, mode))
            if v is None:
                missing.setdefault(xi, []).append((xpos, fail_reason(c, mode), i))
            else:
                xs.append(xpos)
                ys.append(v)
                cs.append(convert.get((c, n), 0.0))
        if mode == "finch":
            # Single bar = COO->Finch level conversion + kernel execution.
            ys = [y + c for y, c in zip(ys, cs)]
            ax.bar(xs, ys, width=width * 0.92, color=colors[i],
                   label="Finch", zorder=3)
        else:
            ax.bar(xs, ys, width=width * 0.92, color=colors[i],
                   label=mode, zorder=3)
    ax.set_yscale("log")
    # Freeze the bar-driven limits, then pin missing-cell markers just above
    # the axis bottom (the bar floor can sit below the log autoscale).
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    mark_y = ymin * 1.5
    for xi, cells in missing.items():
        for xpos, _, i in cells:
            ax.plot([xpos], [mark_y], marker="x", color=colors[i],
                    markersize=6, markeredgewidth=1.6, ls="none",
                    zorder=4, clip_on=False)
        reasons = " / ".join(dict.fromkeys(label for _, label, _ in cells))
        text_x = sum(xpos for xpos, _, _ in cells) / len(cells)
        ax.annotate(reasons, (text_x, mark_y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([CASE_MATH[c] for c in CASE_ORDER],
                       fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("end-to-end time (ms, median)")
    ax.legend(loc="upper left")
    style_axes(ax)
    ax.grid(False, axis="x")
    fig.tight_layout()
    out = out_dir / f"exp2_summary{suffix}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_exp3(results_dir: Path, out_dir: Path) -> None:
    rows = dedupe(
        load_csv(results_dir / "exp3_stages.csv"),
        ("device_mode", "case", "n", "skew", "stage"),
    )
    rows = [r for r in rows if r["status"] == bc.STATUS_OK]
    if not rows:
        print("exp3: no data")
        return
    cases = []
    for r in rows:  # preserve run order
        key = (r["case"], r["skew"])
        if key not in cases:
            cases.append(key)

    fig, axes = plt.subplots(
        2, len(cases), figsize=(3.2 * len(cases), 5.6), squeeze=False
    )
    def has_reduction(eq: str) -> bool:
        ins, out = eq.split("->")
        return bool(set(ins.replace(",", "")) - set(out))

    def round_to_100(shares: list[float]) -> list[int]:
        """Largest-remainder rounding so the labels sum to exactly 100."""
        floors = [int(s) for s in shares]
        rem = 100 - sum(floors)
        order = sorted(range(len(shares)),
                       key=lambda i: shares[i] - floors[i], reverse=True)
        for i in order[:rem]:
            floors[i] += 1
        return floors

    for col, (case, skew) in enumerate(cases):
        crows = [r for r in rows if r["case"] == case and r["skew"] == skew]
        ns = sorted({int(r["n"]) for r in crows})
        med = {
            (int(r["n"]), r["stage"]): float(r["time_ms_median"])
            for r in crows if r["time_ms_median"]
        }
        # A reduction-free einsum never runs merge (the recorded "merge" row
        # is the output-assembly branch, ~µs) — drop the series entirely.
        eq = crows[0]["equation"]
        stages = STAGE_ORDER if has_reduction(eq) else ("mask", "product")
        x = range(len(ns))
        width = 0.26

        # Top: absolute stage times, grouped (log y — stacking would
        # misrepresent the sums on a log axis).
        ax = axes[0][col]
        offset0 = -(len(stages) - 1) / 2
        for i, stage in enumerate(stages):
            ys = [med.get((n, stage)) for n in ns]
            xs = [xi + (offset0 + i) * width for xi in x]
            ax.bar([xi for xi, y in zip(xs, ys) if y is not None],
                   [y for y in ys if y is not None],
                   width=width * 0.92,
                   color=SERIES_COLORS[STAGE_ORDER.index(stage)],
                   label=stage, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{n:,}" for n in ns])
        ax.set_title(CASE_MATH.get(case, case), fontsize=10)
        if col == 0:
            ax.set_ylabel("stage time (ms, median)")
            ax.legend(loc="upper left")
        style_axes(ax)
        ax.grid(False, axis="x")

        # Bottom: share of the stage sum (linear, 100 % stacked). Every
        # segment is labeled; labels use largest-remainder rounding so each
        # bar's numbers sum to exactly 100. Labels that don't fit inside
        # their segment sit beside the bar in the segment's color.
        ax = axes[1][col]
        bar_w = 0.6
        for xi, n in enumerate(ns):
            vals = [med.get((n, s), 0.0) for s in stages]
            total = sum(vals)
            fr = [100.0 * v / total if total else 0.0 for v in vals]
            labels = round_to_100(fr)
            b = 0.0
            last_side_y = -10.0  # stagger side labels ≥ 6 units apart
            for i, stage in enumerate(stages):
                color = SERIES_COLORS[STAGE_ORDER.index(stage)]
                ax.bar([xi], [fr[i]], width=bar_w, bottom=b, color=color,
                       edgecolor="white", linewidth=1.0, zorder=3,
                       label=stage if xi == 0 else None)
                if fr[i] >= 6.0:
                    ax.text(xi, b + fr[i] / 2, f"{labels[i]}%", ha="center",
                            va="center", fontsize=7, color="white")
                else:
                    y = max(b + fr[i] / 2, last_side_y + 6.0)
                    last_side_y = y
                    ax.text(xi + bar_w / 2 + 0.05, y, f"{labels[i]}%",
                            ha="left", va="center", fontsize=7, color=color)
                b += fr[i]
        ax.set_ylim(0, 100)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{n:,}" for n in ns])
        ax.set_xlabel("pieces per operand (n)")
        if col == 0:
            ax.set_ylabel("share of stage sum (%)")
        style_axes(ax)
        ax.grid(False, axis="x")

    fig.tight_layout()
    out = out_dir / "exp3_stages.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", choices=("1", "2", "3", "all"), default="all")
    p.add_argument("--summary-n", type=int, default=None,
                   help="exp2 summary at this n (missing cells annotated); "
                        "default: largest n with full coverage")
    p.add_argument("--results-dir", default=str(bc.RESULTS_DIR))
    p.add_argument("--out-dir", default=str(bc.FIGURES_DIR))
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.exp in ("1", "all"):
        plot_exp1(results_dir, out_dir)
        plot_exp1_summary(results_dir, out_dir, args.summary_n)
    if args.exp in ("2", "all"):
        plot_exp2(results_dir, out_dir)
        plot_exp2_summary(results_dir, out_dir, args.summary_n)
    if args.exp in ("3", "all"):
        plot_exp3(results_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
