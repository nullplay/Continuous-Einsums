# Run the exp2 einsum cases with Finch continuous tensors and append rows to
# results/exp2_finch.csv (same schema as exp2_einsum.csv, device_mode=finch).
#
# Usage:
#   julia --project=benchmarks/finch benchmarks/finch/run_finch_exp2.jl \
#         [--sizes 100,300,...] [--cases a,b] [--data-dir ...] \
#         [--out results/exp2_finch.csv] [--check-only] [--budget 120]
#
# Protocol mirrors bench_common.timed_cell: 2 warmups (the first absorbs any
# residual compilation and enforces the budget), 5 timed reps, median ms.
# Kernel compilation happens once per case via @finch_kernel + eval, before
# any timing. COO->Finch conversion is timed separately (convert_ms in note).

using Dates
using NPZ
using Printf
using Statistics

const HERE = @__DIR__
const BENCH = dirname(HERE)

include(joinpath(HERE, "convert.jl"))
include(joinpath(HERE, "kernels.jl"))

const HEADER = "device_mode,case,equation,n,skew,impl,repeats,warmup,status," *
               "out_nnz,max_rel_diff,time_ms_median,time_ms_all,note,git_rev," *
               "torch_version,polars_version,device_name,thread_info,timestamp"

function parse_args(argv)
    opts = Dict{String,Any}(
        "sizes" => [100, 300, 1000, 3000, 10000, 30000, 100000],
        "cases" => collect(CASES),
        "data-dir" => joinpath(HERE, "data"),
        "out" => joinpath(BENCH, "results", "exp2_finch.csv"),
        "check-only" => false,
        "budget" => 120.0,
        "skew" => 0.5,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--check-only"
            opts["check-only"] = true
        elseif a == "--sizes"
            i += 1; opts["sizes"] = parse.(Int, split(argv[i], ","))
        elseif a == "--cases"
            i += 1; opts["cases"] = String.(split(argv[i], ","))
        elseif a == "--data-dir"
            i += 1; opts["data-dir"] = argv[i]
        elseif a == "--out"
            i += 1; opts["out"] = argv[i]
        elseif a == "--budget"
            i += 1; opts["budget"] = parse(Float64, argv[i])
        elseif a == "--skew"
            i += 1; opts["skew"] = parse(Float64, argv[i])
        else
            error("unknown argument $a")
        end
        i += 1
    end
    return opts
end

git_rev(dir) = try
    readchomp(`git -C $dir rev-parse --short HEAD`)
catch
    ""
end

csv_quote(s::AbstractString) = occursin(r"[,\"]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s

function write_row(io, case, n, skew, status; out_nnz = "", max_rel_diff = "",
                   median_ms = "", all_ms = "", note = "", revs = ("", ""))
    cells = [
        "finch", case, csv_quote(EQUATIONS[case]), string(n), string(skew),
        "ceinsum", "5", "2", status, string(out_nnz), max_rel_diff,
        median_ms, all_ms, csv_quote(note), revs[1], "", "",
        "cpu ($(Sys.CPU_THREADS) cores; julia $(VERSION))",
        "julia_threads=$(Threads.nthreads());finch_rev=$(revs[2])",
        Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "+00:00",
    ]
    println(io, join(cells, ","))
    flush(io)
end

fmt_ms(x) = @sprintf("%.3f", x)

function main()
    opts = parse_args(ARGS)
    data_dir = opts["data-dir"]
    revs = (git_rev(BENCH), git_rev(joinpath(dirname(dirname(BENCH)), "Finch.jl")))
    out_path = opts["out"]
    mkpath(dirname(out_path))
    fresh = !isfile(out_path) || filesize(out_path) == 0
    io = open(out_path, "a")
    fresh && println(io, HEADER)

    for case in opts["cases"]
        println("=== $case ===")
        kernel_ready = false
        kfn = nothing
        compile_ms = 0.0
        failed = false
        for n in sort(opts["sizes"])
            path = joinpath(data_dir, "$(case)_n$(n).npz")
            if !isfile(path)
                println("  n=$n: missing $path — skipped")
                continue
            end
            if failed
                write_row(io, case, n, opts["skew"], "skipped_ladder"; revs = revs)
                continue
            end
            note_parts = String[]
            case == "diagonal_ii_i" && push!(note_parts, "diagonal extracted during conversion")
            try
                raw = npzread(path)
                tconv = @elapsed ((ops, nfrag) = prepare(case, raw))
                push!(note_parts, "convert_ms=$(fmt_ms(tconv * 1e3))")
                nfrag > 0 && push!(note_parts, "nfrag=$nfrag")
                z = make_output(case)

                if !kernel_ready
                    tcomp = @elapsed begin
                        def = kernel_def(case, z, ops)
                        @eval $def
                    end
                    compile_ms = tcomp * 1e3
                    kfn = getfield(Main, kernel_name(case))
                    kernel_ready = true
                end
                push!(note_parts, "compile_ms=$(fmt_ms(compile_ms))")

                # Warmup 1 (budget-checked), warmup 2, then 5 timed reps.
                t1 = @elapsed Base.invokelatest(kfn, z, ops...)
                if t1 > opts["budget"]
                    push!(note_parts, "first_run_s=$(fmt_ms(t1))")
                    write_row(io, case, n, opts["skew"], "budget";
                              median_ms = fmt_ms(t1 * 1e3), all_ms = fmt_ms(t1 * 1e3),
                              note = join(note_parts, ";"), revs = revs)
                    failed = true
                    println("  n=$n: budget ($(round(t1; digits=1)) s)")
                    continue
                end
                Base.invokelatest(kfn, z, ops...)
                times = Float64[]
                res = nothing
                for _ in 1:5
                    t = @elapsed (res = Base.invokelatest(kfn, z, ops...))
                    push!(times, t * 1e3)
                end
                zout = hasproperty(res, :z) ? res.z : z

                max_rel_diff = ""
                ref_path = joinpath(data_dir, "$(case)_n$(n)_ref.npz")
                if isfile(ref_path)
                    ref = npzread(ref_path)
                    ref_total = Float64(vec(ref["ref_total"])[1])
                    fin_total = Base.invokelatest(aggregate, case, zout)
                    rel = abs(fin_total - ref_total) / max(abs(ref_total), 1e-30)
                    max_rel_diff = @sprintf("%.2e", rel)
                    ok = rel <= 1e-3
                    println("  n=$n: check ref=$(ref_total) finch=$(fin_total) rel=$(max_rel_diff) $(ok ? "OK" : "MISMATCH")")
                    ok || push!(note_parts, "CHECK_FAILED")
                end

                if opts["check-only"]
                    isfile(ref_path) || println("  n=$n: no reference file, nothing checked")
                    continue
                end
                write_row(io, case, n, opts["skew"], "ok";
                          out_nnz = stored_entries(zout), max_rel_diff = max_rel_diff,
                          median_ms = fmt_ms(median(times)),
                          all_ms = join(fmt_ms.(times), ";"),
                          note = join(note_parts, ";"), revs = revs)
                println("  n=$n: median $(fmt_ms(median(times))) ms (convert $(fmt_ms(tconv * 1e3)) ms)")
            catch err
                bt = sprint(showerror, err)
                short = first(replace(bt, r"\s+" => " "), 300)
                println("  n=$n: ERROR $short")
                opts["check-only"] || write_row(io, case, n, opts["skew"], "error";
                                                note = join(vcat(note_parts, ["err=" * short]), ";"),
                                                revs = revs)
                failed = true
            end
        end
    end
    close(io)
end

main()
