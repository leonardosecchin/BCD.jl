using DataFrames
using JLD2

include("jld2_read.jl")

results = jld2_read("results.jld2", "results")

# for id in [1;2;3;4], dec in ["dec_min";"dec_max"]
#     print("run_id = $(id), $(dec): ")
# end

for id in [1;2;3;4], dec in ["dec_min";"dec_max"]
    println("\n",'='^15," run_id = $(id), $(dec) ",'='^15)
    rr = results[(results.dec .== dec) .& (results.run_id .== id),[:instance;:st;:sigs]]
    num_problems_inc = 0
    num_inc = 0
    for r in eachrow(rr)
        if r.st == 0
            inc = count(r.sigs .> 1.0)
            println("st $(r.st): \t #>1: $(inc) \tmax: $(maximum(r.sigs)) \t $(r.instance)")
            num_problems_inc += inc > 0
            num_inc += sum(log10.(r.sigs))
        end
    end
    solved = count(rr.st .== 0)
    println("\nNumber of problems solved: ", solved)
    println("% of problems solved where σ increased: $(100*num_problems_inc/solved)%")
    println("Total number of increasements: $(Int64(num_inc))")
    println('='^51)
end
