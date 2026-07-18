using BCD
using DataFrames
using JLD2
using DataFrames
using Plots
using Latexify

include("jld2_read.jl")

results = jld2_read("results.jld2", "results")
results = results[results.st .== 0, :]

table = DataFrame([
    "alpha" => Float64[];
    "dec" => String[];
    "# solved" => Int64[];
    "% prob σ inc" => Float64[];
    "# σ inc" => Int64[]
])

for id in [1;2;3;4;5;6], dec in ["dec_min";"dec_max"]
    rr = results[(results.dec .== dec) .& (results.run_id .== id),:]
    num_problems_inc = 0
    num_inc = 0
    for r in eachrow(rr)
        inc = count(r.sigs .> 1.0)
        num_problems_inc += inc > 0
        num_inc += sum(log10.(r.sigs))
        fig = plot(; title="",
            xlabel="iterations",
            ylabel="",
            fontfamily="Computer Modern"
        )
        ssigs = r.sigs .> 1
        sfs = log.((r.fs .- minimum(r.fs) .+ 1.0)) / log(maximum(r.fs))
#         fig = plot!(1:length(r.sigs), ssigs; label="σ")
#         fig = plot!(1:length(r.fs), sfs; label="f")
#         savefig(fig, "run_$(id)_$(dec)_$(replace(r.instance, "/" => "")).pdf")
    end
    solved = length(rr.st)
    push!(table,
        (rr.alpha, dec, solved, 100*num_problems_inc/solved, Int64(num_inc))
    )
end

display(table)
