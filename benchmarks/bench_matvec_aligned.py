"""One-off: time python ceinsum matvec_ii with cell-aligned A (exp2 protocol).

Usage: python bench_matvec_aligned.py <device-mode>
"""

import sys

import bench_common as bc

bc.apply_device_mode(sys.argv[1])
bc.add_src_to_path()

import torch

from ceinsum_cases import SEED_A, SEED_B, _aligned_interval_ct, _interval_ct
from continuous_einsum import ceinsum

device = bc.apply_device_mode(sys.argv[1])
SKEW = 0.5

for n in (100, 1000, 10000, 100000):
    ops = (
        _aligned_interval_ct(n, SKEW, SEED_A, 2, device),
        _interval_ct(n, SKEW, SEED_B, 1, device),
    )
    nnz = []
    res = bc.timed_cell(
        lambda: ceinsum("ik,k->i", *ops),
        repeats=5, device=device, warmup=2,
        probe=lambda out: nnz.append(out.nnz),
    )
    print(f"{sys.argv[1]} n={n}: {res['status']} median {bc.fmt_ms(res['median_ms'])} ms "
          f"out_nnz={nnz[0] if nnz else '?'}", flush=True)
