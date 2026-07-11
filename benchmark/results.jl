using Plots
using Format
using BenchmarkProfiles
using LaTeXStrings

# run_id: 0 cyclic, 1 metis cyclic

######################
# LATEX TABLE
######################
# Formats
fmt_d = generate_formatter("%'d")
fmt_lf = generate_formatter("%6.2lf")
fmt_lf1 = generate_formatter("%5.1lf")
fmt_e = generate_formatter("%8.2e")
fmt_etex(v) = replace(fmt_e(v), "e+" => "e\$+\$", "e-" => "e\$-\$")

function lplsq_table(; run_id = 0, nb = 10.0)
    results = jld2_read("results.jld2", "results")
    if isnothing(results)
        return
    end
    results = results[(results.run_id .== run_id) .& (results.nb .== nb),:]

    tex = open("lp-lsq_results.tex", "w")
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

function pp_blk(; nb = 10.0, runs = [0;1], p = 1.5)
    results = jld2_read("results.jld2","results")
    results = results[(results.nb .== nb) .& (results.p .== p),:]

    results[results.st .!= 0,:iter] .= -1

    algs = Dict(
        0 => "Cyclic",
        1 => "Cyclic w/ Metis"
    )

    labels = String[]
    iters = []
    for r in runs
        if isempty(iters)
            iters = Float64.(results[results.run_id .== r,:iter])
        else
            iters = hcat(iters, Float64.(results[results.run_id .== r,:iter]))
        end
        push!(labels, algs[r])
    end
    iters[iters .< 0] .= Inf

    fig = performance_profile(PlotsBackend(), iters, labels, title = "Outer iterations", fontfamily="Computer Modern")
    Plots.savefig(fig, "pp_blk_iter.pdf")
end

function pp_S(; nb = 10.0)
    results = jld2_read("results.jld2","results")
    results = results[(results.run_id .== 0) .& (results.p .== 1.5) .& (results.nb .== nb),:]

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