# Per-case operand conversion, Finch kernel definitions, and correctness
# aggregates for the exp2 einsum cases.
#
# Kernel bodies mirror the python equations; a reduced interval index k is
# integrated by multiplying the summand with d(k). Access order is Finch's
# innermost-first (`x[j,i]` = inner j, outer i); "transposed" operands are
# just built with the other dim as the outer level (argument order into
# to_rle_2d).
#
# diagonal_ii_i: Finch cannot lower a repeated index A[i,i] on nested
# continuous levels, so the exact diagonal is extracted during conversion
# (diag_extract) and the kernel is 1D pointwise.

using Finch

const CASES = (
    "pointwise_1d_ii", "pointwise_2d_ii", "diagonal_ii_i", "matvec_ii",
    "matmul_ii", "triple_contract_pii", "box_search",
)

const EQUATIONS = Dict(
    "pointwise_1d_ii" => "i,i->i",
    "pointwise_2d_ii" => "ij,ij->ij",
    "diagonal_ii_i" => "ii,i->i",
    "matvec_ii" => "ik,k->i",
    "matmul_ii" => "ik,kj->ij",
    "triple_contract_pii" => "ij,ik,kj->ij",
    "box_search" => "xy,xy->xy",
)

f32(x) = convert(Vector{Float32}, vec(x))

"Convert one case's npz dict into Finch operands. Returns (ops, nfrag)."
function prepare(case::String, d::Dict)
    if case == "pointwise_1d_ii"
        A = to_rle_1d(f32(d["op0_dim0_start"]), f32(d["op0_dim0_end"]), f32(d["op0_values"]))
        B = to_rle_1d(f32(d["op1_dim0_start"]), f32(d["op1_dim0_end"]), f32(d["op1_values"]))
        return (A, B), 0
    elseif case == "pointwise_2d_ii"
        A, fa = to_rle_2d(f32(d["op0_dim0_start"]), f32(d["op0_dim0_end"]),
                          f32(d["op0_dim1_start"]), f32(d["op0_dim1_end"]), f32(d["op0_values"]))
        B, fb = to_rle_2d(f32(d["op1_dim0_start"]), f32(d["op1_dim0_end"]),
                          f32(d["op1_dim1_start"]), f32(d["op1_dim1_end"]), f32(d["op1_values"]))
        return (A, B), fa + fb
    elseif case == "diagonal_ii_i"
        Dg = diag_extract(f32(d["op0_dim0_start"]), f32(d["op0_dim0_end"]),
                          f32(d["op0_dim1_start"]), f32(d["op0_dim1_end"]), f32(d["op0_values"]))
        B = to_rle_1d(f32(d["op1_dim0_start"]), f32(d["op1_dim0_end"]), f32(d["op1_values"]))
        return (Dg, B), 0
    elseif case == "matvec_ii"
        A, fa = to_rle_2d(f32(d["op0_dim0_start"]), f32(d["op0_dim0_end"]),
                          f32(d["op0_dim1_start"]), f32(d["op0_dim1_end"]), f32(d["op0_values"]))
        B = to_rle_1d(f32(d["op1_dim0_start"]), f32(d["op1_dim0_end"]), f32(d["op1_values"]))
        return (A, B), fa
    elseif case == "matmul_ii"
        A, fa = to_rle_2d(f32(d["op0_dim0_start"]), f32(d["op0_dim0_end"]),
                          f32(d["op0_dim1_start"]), f32(d["op0_dim1_end"]), f32(d["op0_values"]))
        # B is "kj" (dim0=k, dim1=j); build with j as the outer level.
        Bt, fb = to_rle_2d(f32(d["op1_dim1_start"]), f32(d["op1_dim1_end"]),
                           f32(d["op1_dim0_start"]), f32(d["op1_dim0_end"]), f32(d["op1_values"]))
        return (A, Bt), fa + fb
    elseif case == "triple_contract_pii"
        A = to_list_2d(f32(d["op0_dim0_coord"]), f32(d["op0_dim1_coord"]), f32(d["op0_values"]))
        B, fb = to_rle_2d(f32(d["op1_dim0_start"]), f32(d["op1_dim0_end"]),
                          f32(d["op1_dim1_start"]), f32(d["op1_dim1_end"]), f32(d["op1_values"]))
        # C is "kj" (dim0=k, dim1=j); build with j as the outer level.
        Ct, fc = to_rle_2d(f32(d["op2_dim1_start"]), f32(d["op2_dim1_end"]),
                           f32(d["op2_dim0_start"]), f32(d["op2_dim0_end"]), f32(d["op2_values"]))
        return (A, B, Ct), fb + fc
    elseif case == "box_search"
        Box = single_box(f32(d["op0_dim0_start"])[1], f32(d["op0_dim0_end"])[1],
                         f32(d["op0_dim1_start"])[1], f32(d["op0_dim1_end"])[1])
        P = to_list_2d(f32(d["op1_dim0_coord"]), f32(d["op1_dim1_coord"]), f32(d["op1_values"]))
        return (Box, P), 0
    end
    error("unknown case $case")
end

out_rle_1d() = Fiber(SparseRLE{Limit{Float32}}(Element{0.0f0}(), SHAPE))
out_rle_2d() = Fiber(SparseRLE{Limit{Float32}}(
    SparseRLE{Limit{Float32}}(Element{0.0f0}(), SHAPE), SHAPE))
out_list_2d() = Fiber(SparseList{Float32}(
    SparseList{Float32}(Element{0.0f0}(), SHAPE), SHAPE))

function make_output(case::String)
    case in ("pointwise_1d_ii", "diagonal_ii_i", "matvec_ii") && return out_rle_1d()
    case in ("pointwise_2d_ii", "matmul_ii") && return out_rle_2d()
    return out_list_2d()  # triple_contract_pii, box_search
end

"Build the @finch_kernel definition for a case (compile step; eval the result)."
function kernel_def(case::String, z, ops)
    if case == "pointwise_1d_ii"
        A, B = ops
        return @finch_kernel mode=fastfinch function k_pointwise_1d_ii(z, A, B)
            z .= 0
            for i=_; z[i] += A[i] * B[i] end
        end
    elseif case == "pointwise_2d_ii"
        A, B = ops
        return @finch_kernel mode=fastfinch function k_pointwise_2d_ii(z, A, B)
            z .= 0
            for i=_, j=_; z[j, i] += A[j, i] * B[j, i] end
        end
    elseif case == "diagonal_ii_i"
        Dg, B = ops
        return @finch_kernel mode=fastfinch function k_diagonal_ii_i(z, Dg, B)
            z .= 0
            for i=_; z[i] += Dg[i] * B[i] end
        end
    elseif case == "matvec_ii"
        A, B = ops
        return @finch_kernel mode=fastfinch function k_matvec_ii(z, A, B)
            z .= 0
            for i=_, k=_; z[i] += A[k, i] * B[k] * d(k) end
        end
    elseif case == "matmul_ii"
        A, Bt = ops
        return @finch_kernel mode=fastfinch function k_matmul_ii(z, A, Bt)
            z .= 0
            for i=_, j=_, k=_; z[j, i] += A[k, i] * Bt[k, j] * d(k) end
        end
    elseif case == "triple_contract_pii"
        A, B, Ct = ops
        return @finch_kernel mode=fastfinch function k_triple_contract_pii(z, A, B, Ct)
            z .= 0
            for i=_, j=_, k=_; z[j, i] += A[j, i] * B[k, i] * Ct[k, j] * d(k) end
        end
    elseif case == "box_search"
        Box, P = ops
        return @finch_kernel mode=fastfinch function k_box_search(z, Box, P)
            z .= 0
            for x=_, y=_; z[y, x] += Box[y, x] * P[y, x] end
        end
    end
    error("unknown case $case")
end

kernel_name(case::String) = Symbol("k_", case)

"""
Measure-weighted total of the output (the python `ref_total` counterpart):
∫ z dx(dy) for interval outputs, plain value sum for pinpoint outputs.
"""
function aggregate(case::String, z)
    s = Scalar(0.0)
    if case in ("pointwise_1d_ii", "diagonal_ii_i", "matvec_ii")
        @finch (s .= 0; for i=_; s[] += z[i] * d(i) end)
    elseif case in ("pointwise_2d_ii", "matmul_ii")
        @finch (s .= 0; for i=_, j=_; s[] += z[j, i] * d(i, j) end)
    else
        @finch (s .= 0; for i=_, j=_; s[] += z[j, i] end)
    end
    return s.val
end

"Stored output entries (runs/points), informational only."
function stored_entries(z)
    try
        lvl = z.lvl
        while hasproperty(lvl, :lvl) && hasproperty(lvl.lvl, :ptr)
            lvl = lvl.lvl
        end
        return lvl.ptr[end] - 1
    catch
        return -1
    end
end
