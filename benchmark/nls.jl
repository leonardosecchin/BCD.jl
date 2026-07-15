using LinearAlgebra
using SparseArrays
using HSL
using MKL
using Random
using DataFrames
using JLD2
using Metis
using NLSProblems
using NLPModels
using NLPModelsJuMP
using BCD

Random.seed!(0)

struct DATA
    nls::NLPModelsJuMP.MathOptNLSModel
    res::Vector{Float64}
end

function nls_solve(
    nls; q = 10, verbose = 1, usemetis = false, λ = 1.0, p = 1.5, hist = false,
    user_dec = dec_min
)

    n = nls.meta.nvar
    nres = length(residual(nls, zeros(n)))

    function data_initialize(x, bs)
        return DATA(nls, Vector{Float64}(undef, nres))
    end

    function f(x, s, bs, i, data)
        return obj(data.nls, x, data.res, recompute=true) + λ * norm(x, p)^p
    end

    # partial gradient
    function g!(g, x, bs, i, data)
        grad!(data.nls, x, g)
        @inbounds for k in bs[i].idx
            xk = x[k]
#             if abs(xk) > 1e-4
                g[k] += λ*p*abs(xk)^(p-1)*sign(xk)
#             end
        end
    end

    # update B_i
    function B(x, bs, i, data)
        @inbounds H = hess(data.nls, x)[bs[i].idx,bs[i].idx]
        @inbounds ni = bs[i].ni
        @inbounds H += λ * spdiagm(
            ni, ni, p*(p-1)*min.(abs.(x[bs[i].idx]).^(p-2), 10^3)
        )
        return H
    end

    function callback(
        x, blocks::Vector{Block}, iter::IterInfo, par::Param
    )
        push!(fs, iter.f)
        push!(sigs, iter.sig)
        push!(opts, iter.opt)
    end

    if usemetis
        I, J = hess_structure(nls)
        bl_idx = Metis.partition(Symmetric(sparse(I,J,1)), q)
    else
        bl_idx = Vector{Int64}(undef, n)

        # blocks with balanced size
        len, inc = divrem(n, q)
        start = 1
        for i in 1:inc
            bl_idx[start:(start + len)] .= i
            start += len + 1
        end
        for i in (inc + 1):q
            bl_idx[start:(start + len - 1)] .= i
            start += len
        end
    end
    blocks = create_blocks(q, bl_idx)

    par = default_params()
    par.eps = 1e-3
    par.alpha = 1e-4
    par.maxit = 500#max(5000, 100 * q)
    par.maxfnoimpr = ceil(Int64, par.maxit/5)
    par.fest = 1e-5

    x0 = obj(nls, nls.meta.x0) == 0.0 ? ones(n) : nls.meta.x0

    time = @elapsed output = bcd(
        blocks, f, g!, B, data_initialize;
        par = par,
        user_blk = blk_cyclic,
        user_dec = user_dec,
        user_callback = hist ? callback : nothing,
        x0 = x0,
        verbose = verbose
    )

    return output, fs, sigs, opts, time
end

function nls_run_tests(; n = 1000, nb = 10.0)
    problems = [
        n -> mgh21(n), # Rosenbrock (n par)
        n -> mgh22(n), # Powell extended (n must be multiple of 4)
        n -> mgh23(n), # Penalty
        n -> mgh24(n), # Penalty II
        n -> mgh25(n), # Variably dimensioned
        n -> mgh26(n), # Trigonometric function
        n -> mgh27(n), # Brown
        n -> mgh28(n), # Discrete boundary value function
        n -> mgh29(n), # Discrete integral equation function
        n -> mgh30(n), # Broyden tridiagonal
        n -> mgh31(n), # Broyden banded
        n -> mgh32(n), # Linear - full rank
        n -> mgh33(n), # Linear - rank 1
        n -> mgh34(n)  # Linear - rank 1 with zero cols and rows
    ]

    outfile = "results_nls.jld2"

    if isfile(outfile)
        jld2file = jldopen(outfile, "r")
        results = read(jld2file, "results")
        close(jld2file)
    else
        results = DataFrame(
            [
                "instance" => String[]
                "size" => []
                "nblocks" => Int64[]
                "blk_type" => Int64[]
                "f" => Float64[]
                "gsupn" => Float64[]
                "st" => []
                "iter" => Int64[]
                "time" => Float64[]
                "output" => IterInfo[]
                "fs" => Vector{Float64}[]
                "sigs" => Vector{Float64}[]
                "opts" => Vector{Float64}[]
            ]
        )
    end

    nls = []

    for p in problems
#         try
            nls = p(n)
            println("Instance $(nls.meta.name)")

            it, fs, sigs, opts, time = nls_solve(
                nls; q = ceil(Int64, 100 / nb), verbose = 1
            )

            row = [
                nls.meta.name;
                n;
                nb;
                blk_type;
                it.f;
                it.opt;
                it.status;
                it.iter;
                time;
                it;
                [fs];
                [sigs];
                [opts]
            ]
            push!(results, (row))

            jldsave(outfile; results)
#         catch err
#             println("ERROR while solving $(nls.meta.name)")
#         end

        finalize(nls)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    nls_run_tests()
end
