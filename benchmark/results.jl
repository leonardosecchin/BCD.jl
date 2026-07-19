using BCD
using DataFrames
using JLD2
using DataFrames
using Plots
using Format
using BenchmarkProfiles
using LaTeXStrings
using Latexify

include("jld2_read.jl")

######################
# LATEX TABLE
######################
# Formats
fmt_d = generate_formatter("%'d")
fmt_lf = generate_formatter("%6.2lf")
fmt_lf1 = generate_formatter("%5.1lf")
fmt_e = generate_formatter("%8.2e")
fmt_etex(v) = replace(fmt_e(v), "e+" => "e\$+\$", "e-" => "e\$-\$")

function lplsq_table(;
    run_id = 0,
    nb = 10.0,
    dec = "dec_min",
    output = "lp-lsq_results"
)
    results = jld2_read("results.jld2", "results")
    if isnothing(results)
        return
    end
    results = results[
        (results.run_id .== run_id) .&
        (results.nb .== nb) .&
        (results.dec .== dec)
    ,:]

    tex = open("$(output).tex", "w")
    write(tex, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccrrr}\n\\toprule\n")
    write(tex, "Name & \$(m,n)\$ & block size & iter & \$f\$ & opt & time (s)\\\\ \\midrule\n")
    for r in eachrow(results)
        name = replace(basename(string(r.instance)), "_" => "\\_")
        it = (r.gsupn > 1e-3) ? "\\it " : ""
        write(tex, "\\texttt{$(name)} & ($(fmt_d(r.size[1])); $(fmt_d(r.size[2]))) & $(fmt_lf1(nb*r.size[2]/100)) & $(it)$(fmt_d(r.iter)) & $(it)$(fmt_etex(r.f)) & $(it)$(fmt_etex(r.gsupn)) & $(it)$(fmt_lf(r.time)) \\\\ \n")
    end
    write(tex, "\\bottomrule\n\\end{tabular*}")
    close(tex)
end

######################
# PERFORMANCE PROFILES
######################
# replace "powertick" function from BenchmarkProfiles.jl to correctly deal with the Computer Modern font in the x-axis
function BenchmarkProfiles.powertick(s::AbstractString)
    codes = Dict(collect(".1234567890") .=> collect("⋅¹²³⁴⁵⁶⁷⁸⁹⁰"))
    ex = r"[0-9.]+"
    for m ∈ eachmatch(ex, s)
        s = replace(s, m.match => "2^{$(m.match)}")
    end
    return s
end

# function pp_blk(; nb = 10.0, runs = [0;1], p = 1.5)
#     results = jld2_read("results.jld2","results")
#     results = results[(results.nb .== nb) .& (results.p .== p),:]
#
#     results[results.st .!= 0,:iter] .= -1
#
#     algs = Dict(
#         0 => "Cyclic",
#         1 => "Cyclic w/ Metis"
#     )
#
#     labels = String[]
#     iters = []
#     for r in runs
#         if isempty(iters)
#             iters = Float64.(results[results.run_id .== r,:iter])
#         else
#             iters = hcat(iters, Float64.(results[results.run_id .== r,:iter]))
#         end
#         push!(labels, algs[r])
#     end
#     iters[iters .< 0] .= Inf
#
#     fig = performance_profile(PlotsBackend(), iters, labels, title = "Outer iterations", fontfamily="Computer Modern")
#     Plots.savefig(fig, "pp_blk_iter.pdf")
# end

function pp(; nb = 10.0, p = 1.5, run_id = 0)
    results = jld2_read("results.jld2","results")
    results = results[(results.run_id .== run_id) .& (results.p .== p) .& (results.nb .== nb),:]

    results[results.st .!= 0,:time] .= Inf
    results[results.st .!= 0,:iter] .= -1

    S = sort(unique(results[:,:nb]))
    times = []
    iters = []
    labels = String[]
    for ni in S
        if isempty(times)
            times = results[results.nb .== ni,:time]
            iters = Float64.(results[results.nb .== ni,:iter])
        else
            times = hcat(times, results[results.nb .== ni,:time])
            iters = hcat(iters, Float64.(results[results.nb .== ni,:iter]))
        end
        push!(labels, "$(fmt_lf1(ni))%")
    end
    iters[iters .< 0] .= Inf

    fig = performance_profile(PlotsBackend(), times, labels, title = "CPU time", legend = :bottomright, fontfamily="Computer Modern")
    Plots.savefig(fig, "pp_S_time.pdf")

    fig = performance_profile(PlotsBackend(), iters, labels, title = "Outer iterations", legend = :bottomright, fontfamily="Computer Modern")
    Plots.savefig(fig, "pp_S_iter.pdf")
end

function statistics()
    results_all = jld2_read("results.jld2","results")

    if !isdir("figures")
        mkdir("figures")
    end

    # Summarize the results
    table = DataFrame([
        "alpha" => Float64[];
        "dec" => String[];
        "solved" => Int64[];
        "prob_inc" => Int64[];
        "total_inc" => Int64[]
    ])

    results = results_all[(results_all.st .== 0) .& (results_all.run_id .> 0), :]
    ids = unique(results.run_id)
    decs = unique(results.dec)

    for id in ids, dec in decs
        rr = results[(results.dec .== dec) .& (results.run_id .== id),:]
        if isempty(rr)
            println("No problems solved for $(dec), run_id $(id)")
            continue
        end
        num_problems_inc = 0
        num_inc = 0
        for r in eachrow(rr)
            inc = count(r.sigs .> 1.0)
            num_problems_inc += inc > 0
            num_inc += sum(log10.(r.sigs))
#             fig = plot(; title="",
#                 xlabel="iterations",
#                 ylabel="",
#                 fontfamily="Computer Modern"
#             )
#             ssigs = r.sigs .> 1
#             sfs = log.((r.fs .- minimum(r.fs) .+ 1.0)) / log(maximum(r.fs))
#             fig = plot!(1:length(r.sigs), ssigs; label="σ")
#             fig = plot!(1:length(r.fs), sfs; label="f")
#             savefig(fig, "figures/run_$(id)_$(dec)_$(basename(r.instance)).pdf")
        end
        solved = length(rr.st)
        push!(table,
            (rr.alpha[1], dec, solved, num_problems_inc, Int64(num_inc))
        )
    end

    # Write tex table
    tex = open("statistics.tex", "w")
    write(tex, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l$(repeat('c',3*length(decs)))}\n\\toprule\n")
    for dec in decs
        d = replace(basename(string(dec)), "dec_" => "")
        write(tex, " & \\multicolumn{2}{c}{$(d)} &")
    end
    write(tex, "\\\\ \n \$\\alpha\$")
    for dec in decs
        write(tex, " & \\#sol. & \\#\$\\sigma\\uparrow\$ &")
    end
    write(tex, "\\\\ \\midrule\n")

    alphas = unique(table.alpha)
    for a in alphas
        write(tex, "\$10^{$(Int64(log10(a)))}\$")
        for dec in decs
            t = table[(table.dec .== dec) .& (table.alpha .== a),:]
            if length(t[:,1]) != 1
                write(tex, "& -- & -- &")
                continue
            end
            write(tex, "& $(t.solved[1]) ($(t.prob_inc[1])) & $(t.total_inc[1]) &")
        end
        write(tex, "\\\\ \n")
    end
    write(tex, "\\bottomrule\n\\end{tabular*}")
    close(tex)

    # Figures of each problem, specific run_id
    decs = ["dec_min";"dec_max";"dec_onlyE"]
    results = results_all[(results_all.run_id .== 6) .& (results_all.st .== 0), :]

    for p in eachrow(results)
        name = basename(p.instance)
        fig = plot(; title=name,
            xlabel="iterations",
            ylabel="times increased",
            fontfamily="Computer Modern"
        )
        for dec in decs
            rr = results[(results.instance .== p.instance) .& (results.dec .== dec), :]
            if isempty(rr)
                continue
            end
            sigs = log10.(rr.sigs[1])
            fig = plot!(1:length(sigs), sigs, label=dec)
            println("Problem $(name), $(dec), times increased: $(sum(sigs))")
        end
        savefig(fig, "figures/sigma_$(name).pdf")
    end

#     data = zeros(Int64, length(results.instance), length(decs))
#     for p in 1:length(results.instance)
#         for d in 1:length(decs)
#             rr = results[(results.instance .== results.instance[p]) .& (results.dec .== decs[d]), :]
#             if isempty(rr)
#                 continue
#             end
#             data[p,d] = sum(log10.(rr.sigs[1]))
#         end
#     end

#     fig = bar(
#         basename.(results.instance),
#         data;
#         bar_position = :dodge,
#         label = decs,
#         xrotation = 90,
#         size = (1600, 600),
#         bar_width = 0.8
#     )
#     savefig(fig, "teste.pdf")
end
