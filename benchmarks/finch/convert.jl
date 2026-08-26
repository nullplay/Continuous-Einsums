# COO piece data -> Finch continuous level formats.
#
# Python exp2 operands are COO "pieces": per dim either a half-open interval
# [start, end) or a pinpoint coordinate. Finch continuous levels are CSF-like
# (SparseRLE / SparseList nests) and need each fiber's runs sorted and
# disjoint, so 2D interval operands are fragmented at x-endpoint breakpoints
# (pieces live strictly inside distinct grid cells, so only same-column
# pieces ever overlap in x and the global breakpoint grid is exact).
#
# Half-open [s, e) becomes the closed Finch run [s, prevfloat(e)]: abutting
# cell-aligned runs stay disjoint and boundary pinpoints stay excluded, at a
# cost of 1 ulp per endpoint.
#
# Finch continuous extents span (1, shape) — make_extent(Ti, 1, shape) — so
# coordinates below 1.0 would be clipped (finch_spatial.jl offsets its data
# for the same reason). Every coordinate is shifted by OFFSET = +1; integrals
# are translation-invariant, so results stay comparable to python.

using Finch

const OFFSET = 1.0f0
const SHAPE = 64.0f0 + OFFSET

"1D interval pieces are naturally disjoint: sort by start."
function to_rle_1d(s::Vector{Float32}, e::Vector{Float32}, v::Vector{Float32};
                   shape::Float32 = SHAPE)
    p = sortperm(s)
    n = length(v)
    Fiber(SparseRLE{Float32}(Element{0.0f0}(v[p]), shape, [1, n + 1],
                             s[p] .+ OFFSET, prevfloat.(e[p] .+ OFFSET)))
end

"""
2D interval pieces -> nested SparseRLE via x-axis fragmentation.

Outer dim is the first coordinate pair passed (callers transpose by argument
order). Returns `(fiber, nfrag)`.
"""
function to_rle_2d(xs::Vector{Float32}, xe::Vector{Float32},
                   ys::Vector{Float32}, ye::Vector{Float32},
                   v::Vector{Float32}; shape::Float32 = SHAPE)
    n = length(v)
    xs = xs .+ OFFSET
    xe = xe .+ OFFSET
    ys = ys .+ OFFSET
    ye = ye .+ OFFSET
    bp = sort!(unique(vcat(xs, xe)))
    t0 = Vector{Int}(undef, n)
    t1 = Vector{Int}(undef, n)
    total = 0
    @inbounds for p in 1:n
        a = searchsortedfirst(bp, xs[p])
        b = searchsortedfirst(bp, xe[p]) - 1
        t0[p] = a
        t1[p] = b
        total += b - a + 1
    end
    fseg = Vector{Int32}(undef, total)
    fys = Vector{Float32}(undef, total)
    fye = Vector{Float32}(undef, total)
    fv = Vector{Float32}(undef, total)
    k = 0
    @inbounds for p in 1:n, t in t0[p]:t1[p]
        k += 1
        fseg[k] = t
        fys[k] = ys[p]
        fye[k] = ye[p]
        fv[k] = v[p]
    end
    perm = sortperm(collect(zip(fseg, fys)))

    inner_s = fys[perm]
    inner_e = prevfloat.(fye[perm])
    inner_v = fv[perm]
    outer_left = Float32[]
    outer_right = Float32[]
    ptr = Int[1]
    prev = Int32(-1)
    @inbounds for (q, idx) in enumerate(perm)
        t = fseg[idx]
        if t != prev
            if prev != -1
                push!(ptr, q)
            end
            push!(outer_left, bp[t])
            push!(outer_right, prevfloat(bp[t + 1]))
            prev = t
        end
    end
    push!(ptr, total + 1)
    nseg = length(outer_left)
    fiber = Fiber(SparseRLE{Float32}(
        SparseRLE{Float32}(Element{0.0f0}(inner_v), shape, ptr, inner_s, inner_e),
        shape, [1, nseg + 1], outer_left, outer_right))
    return fiber, total
end

"2D pinpoints -> nested SparseList (same-column x coords are bitwise equal)."
function to_list_2d(xc::Vector{Float32}, yc::Vector{Float32}, v::Vector{Float32};
                    shape::Float32 = SHAPE)
    n = length(v)
    xc = xc .+ OFFSET
    yc = yc .+ OFFSET
    perm = sortperm(collect(zip(xc, yc)))
    xs = xc[perm]
    ys = yc[perm]
    vs = v[perm]
    outer_x = Float32[]
    ptr = Int[1]
    prev = NaN32
    @inbounds for q in 1:n
        if xs[q] != prev
            if !isempty(outer_x)
                push!(ptr, q)
            end
            push!(outer_x, xs[q])
            prev = xs[q]
        end
    end
    push!(ptr, n + 1)
    nx = length(outer_x)
    Fiber(SparseList{Float32}(
        SparseList{Float32}(Element{0.0f0}(vs), shape, ptr, ys),
        shape, [1, nx + 1], outer_x))
end

"""
Exact `ii` diagonal of 2D interval pieces: per piece the segment
[max(xs,ys), min(xe,ye)) where nonempty. Survivors are pairwise disjoint
(each lives in a distinct diagonal cell).
"""
function diag_extract(xs::Vector{Float32}, xe::Vector{Float32},
                      ys::Vector{Float32}, ye::Vector{Float32},
                      v::Vector{Float32}; shape::Float32 = SHAPE)
    lo = max.(xs, ys)
    hi = min.(xe, ye)
    keep = lo .< hi
    to_rle_1d(lo[keep], hi[keep], v[keep]; shape = shape)
end

# Single query box, value 1.0 like the python operand. Built as a 1-run
# nested SparseRLE: the SingleRLE(Pattern()) form from finch_spatial.jl
# miscounts here (it emits phantom entries at the box's right endpoint), so
# we stay on the SparseRLE path the continuous test suite exercises.
function single_box(xs::Float32, xe::Float32, ys::Float32, ye::Float32;
                    shape::Float32 = SHAPE)
    Fiber(SparseRLE{Float32}(
        SparseRLE{Float32}(Element{0.0f0}([1.0f0]), shape, [1, 2],
                           [ys + OFFSET], [prevfloat(ye + OFFSET)]),
        shape, [1, 2], [xs + OFFSET], [prevfloat(xe + OFFSET)]))
end
