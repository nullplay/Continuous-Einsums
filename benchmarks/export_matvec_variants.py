"""One-off: export matvec_ii operand variants to isolate the Finch slowdown.

Variant "aligned":  A cell-aligned (_aligned_interval_ct), B as in exp2.
Variant "bigB":     A as in exp2 (unaligned), B a single run over [0, 64).

Files land in finch/data_{variant}/matvec_ii_n{n}.npz so the unmodified
Julia runner picks them up via --data-dir. Refs (python ceinsum totals)
written for every exported size so the Julia check runs everywhere.
"""

from __future__ import annotations

from pathlib import Path

import bench_common as bc

bc.apply_device_mode("cpu-multi")
bc.add_src_to_path()

import numpy as np
import torch

from ceinsum_cases import (
    SEED_A, SEED_B, SPACE_MAX, DTYPE,
    _aligned_interval_ct, _interval_ct,
)
from ctensor import continuous_tensor
from continuous_einsum import ceinsum
from export_finch_data import tensor_arrays, output_total

SIZES = [100, 1000, 10000, 100000]
SKEW = 0.5
REF_MAX_N = 100000
HERE = Path(__file__).resolve().parent
device = torch.device("cpu")


def single_run_b():
    return continuous_tensor(
        [(torch.tensor([0.0], dtype=DTYPE), torch.tensor([SPACE_MAX], dtype=DTYPE))],
        torch.ones(1, dtype=DTYPE),
        ["[)"],
    )


def make(variant: str, n: int):
    if variant == "aligned":
        return (
            _aligned_interval_ct(n, SKEW, SEED_A, 2, device),
            _interval_ct(n, SKEW, SEED_B, 1, device),
        )
    if variant == "bigB":
        return (_interval_ct(n, SKEW, SEED_A, 2, device), single_run_b())
    raise ValueError(variant)


for variant in ("aligned", "bigB"):
    out_dir = HERE / "finch" / f"data_{variant}"
    out_dir.mkdir(exist_ok=True)
    for n in SIZES:
        ops = make(variant, n)
        arrays = {
            "meta_nops": np.asarray([len(ops)], dtype=np.int64),
            "meta_n": np.asarray([n], dtype=np.int64),
            "space_max": np.asarray([64.0], dtype=np.float64),
        }
        for o, op in enumerate(ops):
            arrays.update(tensor_arrays(op, o))
        np.savez(out_dir / f"matvec_ii_n{n}.npz", **arrays)

        if n <= REF_MAX_N:
            out = ceinsum("ik,k->i", *ops)
            np.savez(
                out_dir / f"matvec_ii_n{n}_ref.npz",
                ref_total=np.asarray([output_total(out)], dtype=np.float64),
                ref_out_nnz=np.asarray([out.nnz], dtype=np.int64),
            )
            print(f"{variant} n={n}: exported, ref_total={output_total(out):.6g}, "
                  f"out_nnz={out.nnz}")
        else:
            print(f"{variant} n={n}: exported")
