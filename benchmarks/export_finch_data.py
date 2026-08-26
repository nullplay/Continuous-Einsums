"""Export the exp2 synthetic operands (and reference aggregates) for Finch.

Dumps every case's operands, regenerated with the exact exp2 seeds, to
``.npz`` files that the Julia runner (``finch/run_finch_exp2.jl``) reads.
For ``--ref-sizes`` it additionally runs the python ``ceinsum`` once and
stores the output's total measure-weighted sum, the correctness reference
the Finch side checks against.

Keys per ``{case}_n{n}.npz`` (all numeric so NPZ.jl round-trips cleanly):

* ``op{o}_dim{d}_start`` / ``op{o}_dim{d}_end``  — interval dim (float32)
* ``op{o}_dim{d}_coord``                          — pinpoint dim (float32)
* ``op{o}_values``                                — float32
* ``op{o}_props``                                 — int8, 0=interval 1=pinpoint
* ``meta_nops``, ``meta_n``, ``space_max``

Per ``{case}_n{n}_ref.npz``: ``ref_total`` (float64 scalar,
``sum(value * prod(interval widths))`` of the ceinsum output; pinpoint dims
contribute width 1) and ``ref_out_nnz`` (int64, informational).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bench_common as bc

bc.apply_device_mode("cpu-multi")
bc.add_src_to_path()

import numpy as np
import torch

from ceinsum_cases import CEINSUM_CASES  # noqa: E402  (torch env set above)

DEFAULT_SIZES = "100,300,1000,3000,10000,30000,100000"
DEFAULT_REF_SIZES = "100,1000"


def tensor_arrays(op, o: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    props = []
    for d, (spec, prop) in enumerate(zip(op.dims, op.property)):
        if prop == "P":
            out[f"op{o}_dim{d}_coord"] = spec[0].cpu().numpy()
            props.append(1)
        else:
            out[f"op{o}_dim{d}_start"] = spec[0].cpu().numpy()
            out[f"op{o}_dim{d}_end"] = spec[1].cpu().numpy()
            props.append(0)
    out[f"op{o}_values"] = op.values.cpu().numpy()
    out[f"op{o}_props"] = np.asarray(props, dtype=np.int8)
    return out


def output_total(out) -> float:
    """Measure-weighted sum of a ceinsum output, in float64."""
    total = out.values.cpu().double()
    for spec, prop in zip(out.dims, out.property):
        if prop != "P":
            total = total * (spec[1].cpu().double() - spec[0].cpu().double())
    return float(total.sum().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--ref-sizes", default=DEFAULT_REF_SIZES)
    ap.add_argument("--skew", type=float, default=0.5)
    ap.add_argument("--cases", default="")
    ap.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent / "finch" / "data")
    )
    args = ap.parse_args()

    sizes = bc.parse_int_list(args.sizes)
    ref_sizes = set(bc.parse_int_list(args.ref_sizes))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s for s in args.cases.split(",") if s.strip()}
    device = torch.device("cpu")

    from continuous_einsum import ceinsum

    for case in CEINSUM_CASES:
        if wanted and case.label not in wanted:
            continue
        for n in sorted(set(sizes) | ref_sizes):
            if n not in sizes and n not in ref_sizes:
                continue
            ops = case.make_operands(n, args.skew, device)
            arrays: dict[str, np.ndarray] = {
                "meta_nops": np.asarray([len(ops)], dtype=np.int64),
                "meta_n": np.asarray([n], dtype=np.int64),
                "space_max": np.asarray([64.0], dtype=np.float64),
            }
            for o, op in enumerate(ops):
                arrays.update(tensor_arrays(op, o))
            path = out_dir / f"{case.label}_n{n}.npz"
            np.savez(path, **arrays)
            print(f"wrote {path}")

            if n in ref_sizes:
                out = ceinsum(case.equation, *ops)
                ref_path = out_dir / f"{case.label}_n{n}_ref.npz"
                np.savez(
                    ref_path,
                    ref_total=np.asarray([output_total(out)], dtype=np.float64),
                    ref_out_nnz=np.asarray([out.nnz], dtype=np.int64),
                )
                print(f"wrote {ref_path} (total={output_total(out):.9g}, nnz={out.nnz})")


if __name__ == "__main__":
    main()
