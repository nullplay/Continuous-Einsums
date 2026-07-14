# Continuous Einsum via Interaction Tables

## 1. Continuous tensors

A normal tensor is indexed by integers: `A[2,7] = 3.0`. A **continuous tensor**
is indexed by the real line: each dimension's coordinate is a *point*
(`t = 0.5`) or an *interval* (`[1.0, 3.0)`).

Like a sparse tensor in COO format, a continuous tensor is stored as a finite
list of **pieces**. A piece fixes a coordinate for every dimension and carries
one value; everywhere not covered by a piece, the tensor is zero. We write

```
Xd[p]     the coordinate of piece p in dimension d of tensor X
Xval[p]   the value of piece p
```

For now, every coordinate is a point ("pinpoint"); intervals enter in §6.

## 2. Continuous einsum: the one-word change

The discrete einsum `C[i,j] = Σ_k A[i,k] * B[k,j]` combines entries whose
shared indices are **equal**, multiplies their values, and sums over the
contracted index. Continuous einsum keeps the recipe and changes one word:

> entries combine when shared indices are **equal** → pieces combine when
> shared coordinates **intersect**.

For points, "intersect" is still equality — so the pinpoint case looks exactly
like sparse-tensor algebra, and it is the right place to build the machinery.

## 3. The four-step pipeline, on matrix multiply

Take `C[i,j] = Σ_k A[i,k] * B[k,j]` with three pieces per input:

```
A: piece   A0[a] (i)   A1[a] (k)   Aval[a]      B: piece   B0[b] (k)   B1[b] (j)   Bval[b]
   a0      1           0.5         2               b0      0.5         7           10
   a1      1           2.5         3               b1      2.5         7           20
   a2      4           0.5         5               b2      9           8           30
```

### Step 1 — Input interaction table

Record which pieces can combine, as a boolean table with one axis per input.
The only index shared between the inputs is `k`, so:

```
Table[a,b] = (A1[a] == B0[b])

            b0   b1   b2
      a0  [  1    0    0 ]        a0–b0 and a2–b0 meet at k=0.5,
      a1  [  0    1    0 ]        a1–b1 meets at k=2.5
      a2  [  1    0    0 ]
```

This is an adjacency matrix: nodes are pieces, edges are "these interact".

### Step 2 — Output coordinate construction

Before computing anything, list the coordinates the output could possibly
have. An output `i` can only be a value appearing in `A0`; an output `j` only
a value in `B1`. Enumerate all combinations:

```
(Ocrd0, Ocrd1) = flatten( unique(A0) × unique(B1) )
               = flatten( (1,4) × (7,8) )

   o     Ocrd0[o] (i)   Ocrd1[o] (j)
   o0    1              7
   o1    1              8
   o2    4              7
   o3    4              8
```

These `m = 4` **candidate pieces** are the output's coordinate arrays,
finished before any value is touched. Some candidates will turn out to hold
value 0 — that is fine; they are ordinary (explicitly stored zero) pieces.

### Step 3 — Table extension

The table relates inputs to each other but says nothing about *where* each
product lands. Fix that by joining the candidates in, with the same kind of
condition used in step 1:

```
Table+[o,a,b] = Table[a,b] && (Ocrd0[o] == A0[a]) && (Ocrd1[o] == B1[b])

nonzeros:   (o0, a0, b0)   (o0, a1, b1)   (o2, a2, b0)
```

The output behaves exactly like one more operand: it carries `i` and `j`, so
it gets one condition against each input that also carries them.

### Step 4 — Value computation

All geometry is now inside `Table+`; the values follow by plain
multiply-and-sum:

```
Outval[o] = Σ_{a,b}  Table+[o,a,b] * Aval[a] * Bval[b]

Outval = ( 2*10 + 3*20,  0,  5*10,  0 ) = ( 80, 0, 50, 0 )
```

The two tuples landing on `o0` and being summed — that *is* the contraction
over `k`. The result is the piece list `(Ocrd0, Ocrd1, Outval)`:

```
   o     i    j    value
   o0    1    7    80
   o1    1    8    0
   o2    4    7    50
   o3    4    8    0
```

Notice what step 4 is: a **discrete einsum**. The continuous equation
`ik,kj->ij` over coordinates became `oab,a,b->o` over piece numbers.

## 4. The recipe for an arbitrary equation

Let the equation be `X1[s1], …, XN[sN] -> Y[s0]`, where each `sn` is an index
string (e.g. `s1 = "ik"`). Say a tensor **carries** an index letter `x` in
dimension `d` when `sn[d] = x`; the **carriers** of `x` are all such `(Xn, d)`
pairs, and the output `Y` counts as a carrier of the letters in `s0`.

```
Step 1 — Input interaction table                       (one axis per input)
    Table[p1,…,pN] = AND over every index letter x,
                     AND over every pair of input carriers (Xm,d), (Xn,d') of x:
                         Xm_d[pm] == Xn_d'[pn]

Step 2 — Output coordinate construction               (the output's piece list)
    for each output dimension d with letter x = s0[d]:
        Cand(x) = unique( coordinates of the input carriers of x )
    (Ycrd_1,…,Ycrd_r) = flatten( Cand(s0[1]) × … × Cand(s0[r]) ),   o = 1..m

Step 3 — Table extension                               (join the output in)
    Table+[o,p1,…,pN] = Table[p1,…,pN]
                        AND over every output dimension d with letter x,
                        AND over every input carrier (Xn,d') of x:
                            Ycrd_d[o] == Xn_d'[pn]

Step 4 — Value computation                             (a discrete einsum)
    Yval[o] = Σ over (p1,…,pN) of  Table+[o,p1,…,pN] · X1val[p1] · … · XNval[pN]
```

Remarks:

- **Steps 1 and 3 are one rule.** Treat `Y` as tensor number 0; then
  `Table+[o,p1,…,pN] = 1` iff *for every index letter, all of its carriers
  agree*. Step 1 is the input–input part of that rule, step 3 the
  output–input part.
- **All pairs vs. a chain.** With equality it suffices to compare carriers in
  a chain (equality is transitive), so many of the pairwise conditions are
  redundant. Writing "all pairs" costs nothing and is the form that stays
  correct for intervals (§6), where intersection is *not* transitive.
- **Arbitrary index patterns come for free.** A repeated letter inside one
  tensor (`ii->i`, the diagonal) makes that tensor carry the letter twice, and
  the rule emits `X_0[p] == X_1[p]` — a self-condition selecting diagonal
  pieces. Contracted letters (`k` above) are simply letters with no output
  carrier: they generate conditions in step 1 and nothing in step 3, and the
  sum in step 4 eliminates them.
- **The table has N+1 axes** — one per input, sized by its piece count, plus
  one for the output, sized by the candidate count `m`. Step 4 is the discrete
  einsum `"o p1 … pN, p1, …, pN -> o"` between the table and the value lists.

## 5. The freedom in step 2

Step 2 is the only step with a choice in it. Any candidate list works as long
as it (a) contains every coordinate combination that can actually receive a
product and (b) has no duplicates. Three constructions, from loosest to
tightest:

1. **Grid**: cross product of per-letter candidate sets, as above. Always
   available; `m` can be large, and the extra rows just hold zeros.
2. **Zip**: if a single input carries *all* the output letters, its own piece
   list (deduplicated) is a complete candidate list. Example:
   `C[i,j] = Σ_k A[i,k]·B[k,j]·D[i,j]` — every surviving tuple must agree
   with some piece of `D` on both `i` and `j`, so the candidates are just
   `unique_pairs(D0, D1)`. (`D` acts as a mask; this equation is the sampled
   matmul, and the output pattern is at most `D`'s pattern.)
3. **Harvest**: run step 1 first, project its nonzero tuples onto the output
   letters, deduplicate. This is the exact realized set — the tightest
   possible — at the cost of making step 2 depend on step 1. (For sparse
   matmul this is precisely the classical "symbolic phase".)

Whether a tight candidate list is knowable *before* any joining is a property
of the equation's shape: it is when some operand carries all output letters,
and it is not when the output letters are scattered (in `ik,kj->ij`, which `i`
co-occurs with which `j` is created by the contraction itself).

## 6. From points to intervals

Everything so far survives with two substitutions.

**Substitution 1 — the predicate.** "Coordinates agree" becomes "coordinates
intersect", by the type of the pair:

```
point, point           c == c'
point, interval        s <= c  &&  c < e            (for [s,e); brackets
interval, interval     s < e'  &&  s' < e            toggle < vs <=)
```

**Substitution 2 — step 2's `unique` becomes *cutting*.** An output region is
an intersection of carrier intervals, so its endpoints are inherited from
carrier endpoints. Collect every endpoint of every input carrier of the
letter, sort, deduplicate; the candidate coordinates are the **cells** —
segments between consecutive endpoints. The cells are disjoint and every
realized output region is exactly a union of cells.

The reason step 3 stays valid is the

> **grid lemma** — a cell contains no carrier endpoint in its interior, so a
> cell either lies entirely inside or entirely outside any carrier interval;
> hence *overlapping* every carrier is the same as being *contained* in their
> intersection.

That is why the pairwise-intersection rule can keep playing the role that
pairwise equality played — but only because the cells were cut at *all*
carriers' endpoints, and only with the conditions against *all* carriers
present (intersection is not transitive: a cell can overlap `A`'s interval
yet miss `B`'s).

One genuinely new behaviour appears: a tuple's deposit region can span
*several* cells, so one tuple sets several `o`-bits in `Table+`. Take
`C[i] = Σ_j A[i,j]·B[i]·D[j]` with

```
A:  a0 (i:[0,2), j:[0,1), v=2)    a1 (i:[1,3), j:[5,6), v=3)
B:  b0 (i:[1,4], v=10)            b1 (i:[10,12], v=20)
D:  d0 (j: point 0.5, v=100)      d1 (j: point 5.5, v=200)
```

Step 1 leaves two tuples: `(a0,b0,d0)` and `(a1,b0,d1)`. Step 2 cuts `i` at
`{0,1,2,3} ∪ {1,4,10,12}`, giving cells
`[0,1) [1,2) [2,3) [3,4) [4,10) [10,12)`. In step 3, tuple `(a0,b0,d0)`
(deposit region `[0,2)∩[1,4] = [1,2)`) hits the cell `[1,2)`; tuple
`(a1,b0,d1)` (region `[1,3)`) hits `[1,2)` *and* `[2,3)`:

```
deposits:        [ 2000 )
                 [     6000     )
cells:      [0,1)[1,2) [2,3)[3,4)[4,10)[10,12)
Yval:         0  8000  6000   0    0     0
```

The overlap between the two deposit regions never becomes a problem: the
cells were disjoint before any value was computed. (In piece-list evaluations
that compute output pieces per tuple, this same situation is what forces a
separate "coalesce" pass afterwards; here it is absorbed into step 2's
choice of candidates.)

Fine print: with closed brackets two intervals can intersect in a single
point (`[0,2] ∩ [2,5] = {2}`), so full generality asks the cut to also emit
zero-width cells at closed endpoints. The rules are unchanged; only the
candidate list grows.

## 7. Summary

A continuous einsum factors into structure times arithmetic:

1. **Input interaction table** — which input pieces combine (adjacency
   tensor, one axis per input; every index letter demands agreement between
   each pair of its input carriers).
2. **Output coordinate construction** — the output's coordinate arrays,
   enumerated from the inputs' coordinates (unique values for points, cells
   between endpoints for intervals).
3. **Table extension** — the output joins the table as an (N+1)-th operand,
   under the same per-letter agreement rule.
4. **Value computation** — one discrete einsum between the table and the
   value lists.

Steps 1–3 touch only coordinates; step 4 touches only values. Setting every
coordinate to an integer point collapses "intersect" to "equal" and the
construction reproduces ordinary sparse einsum exactly: continuous einsum is
a strict generalization. The uncompressed table is exponential in the number
of operands, so it is best read not as an algorithm but as the
*specification* that any practical evaluation strategy must agree with.

The optimized pipeline (`continuous_einsum.ceinsum`) is that practical
strategy: it realizes the same specification as a sparse **mask → product →
merge** computation — the interaction table stored sparsely as a join
(`ceinsum_mask`), per-tuple values and coordinates (`ceinsum_product`), and a
discrete merge plus sweep-line coalesce (`ceinsum_merge`) — and it shares the
integral semantics above: a contracted all-interval index weights each
contribution by its overlap length.
