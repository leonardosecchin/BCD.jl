using HSL
using BCD
using LinearAlgebra
using SparseArrays
using MKL
using Pkg
using Random
using DataFrames
using JLD2
using Metis
using MatrixDepot

include("jld2_read.jl")
include("results.jl")

Random.seed!(0)

matrices = [
"Priebel/162bit",
"Priebel/176bit",
"JGD_Homology/ch6-6-b3",
"JGD_Homology/ch7-6-b3",
"JGD_Homology/ch7-8-b2",
"JGD_Homology/ch7-9-b2",
"JGD_Homology/cis-n4c6-b3",
"JGD_Franz/Franz4",
"JGD_Franz/Franz5",
"JGD_Franz/Franz6",
"JGD_Franz/Franz7",
"JGD_Franz/Franz8",
"JGD_Franz/Franz9",
"JGD_Franz/Franz10",
"JGD_GL7d/GL7d12",
"Kemelmacher/Kemelmacher",
"NYPA/Maragal_5",
"NYPA/Maragal_4",
"JGD_Homology/mk10-b3",
"JGD_Homology/mk11-b3",
"JGD_Homology/mk12-b2",
"JGD_Homology/n2c6-b4",
"JGD_Homology/n2c6-b5",
"JGD_Homology/n2c6-b6",
"JGD_Homology/n3c6-b4",
"JGD_Homology/n3c6-b5",
"JGD_Homology/n3c6-b6",
"JGD_Homology/n4c5-b4",
"JGD_Homology/n4c5-b5",
"JGD_Homology/n4c5-b6",
"JGD_Homology/n4c6-b3"
]

struct DATA
    A::Vector{SparseMatrixCSC{Float64, Int64}}
    Axb::Vector{Float64}
    Asi::Vector{Float64}
    B::Vector{SparseMatrixCSC{Float64, Int64}}
    b::Vector{Float64}
end

function download_matrices()
    problems = DataFrame(
        [
            "name" => String[]
            "A" => SparseMatrixCSC{Float64, Int64}[]
            "b" => Vector{Float64}[]
        ]
    )
    for m in matrices
        M = mdopen(m)
        A = deepcopy(M.A)
        b = rand(size(A,1))
        push!(problems, [m, A, b])
    end
    jldsave("lp-lsq.jld2"; problems)
end

function solve(
    A, b;
    q = 10,
    p = 1.5,
    usemetis = false,
    x0 = [],
    user_blk = blk_cyclic,
    user_dec = user_dec,
    verbose = 1
)
    @assert length(b) == size(A,1) "Dimensions mismatch"

    n = size(A, 2)

    D = max(1.0/norm(A,2)^2, 10^3)

    # function to allocate and initialize data
    function data_initialize(x, bs)
        data = DATA(
            SparseMatrixCSC{Float64, Int64}[],
            zeros(length(b)),
            Vector{Float64}(undef, size(A,1)),
            SparseMatrixCSC{Float64, Int64}[],
            b
        )
        @inbounds for i in eachindex(bs)
            push!(data.A, sparse(A[:,bs[i].idx]))
            data.Axb .+= data.A[i] * x[bs[i].idx]
            push!(data.B, tril(transpose(data.A[i]) * data.A[i]))
            dropzeros!(data.B[end])
        end
        data.Axb .-= data.b
        return data
    end

    # f(x) = 1/p |Ax - b|_p^p
    # i = 0 indicates that data.Axb is up to date, so we compute the p-norm
    # directly
    function f(s, bs, i, data)
        if (i > 0)
            # compute Ai * s
            @inbounds data.Asi .= data.A[i] * s
            # update A*x - b
            @. data.Axb += data.Asi
        end
        return (1/p) * norm(data.Axb, p)^p
    end

    # partial gradient
    function g!(g, bs, i, data)
        @inbounds @views if p == 2.0
            g[bs[i].idx] .= transpose(data.A[i]) * data.Axb
        else
            g[bs[i].idx] .= transpose(data.A[i]) * (abs.(data.Axb).^(p-1) .* sign.(data.Axb))
        end
    end

    # update B_i
    function B(bs, i, data)
        @inbounds ni = bs[i].ni
        @inbounds if p == 2.0
            return transpose(data.A[i]) * data.A[i]
        else
            return (p-1)*transpose(data.A[i]) * spdiagm(min.(abs.(data.Axb).^(p-2), D)) * data.A[i]
        end
    end

    function Id(bs, i, data)
        @inbounds ni = bs[i].ni
        @inbounds spdiagm(ni, ni, ones(ni))
    end

    if usemetis
        H = A'*A
        dropzeros!(H)
        bl_idx = Metis.partition(H, nb)
    else
        bl_idx = Vector{Int64}(undef, n)

        # consecutive blocks with balanced size
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
    par.maxit = max(5000, 100 * q)
    par.maxfnoimpr = ceil(Int64, par.maxit/5)

    time = @elapsed output = bcd(
        blocks, f, g!, B, data_initialize;
        par = par,
        user_blk = user_blk,
        user_dec = user_dec,
        x0 = x0,
        verbose = verbose
    )

    return output, time
end

function run_tests(;
    p = 1.5,
    nb = 0.5,
    run_id = 0,
    usemetis = false,
    user_blk = blk_cyclic,
    user_dec = dec_min
)
    outfile = "results.jld2"

    if isfile(outfile)
        jld2file = jldopen(outfile, "r")
        results = read(jld2file, "results")
        close(jld2file)
    else
        results = DataFrame(
            [
            "run_id" => Int64[]
            "instance" => String[]
            "size" => []
            "nb" => Float64[]
            "nblocks" => Int64[]
            "p" => Float64[]
            "blk" => String[]
            "dec" => String[]
            "f" => Float64[]
            "gsupn" => Float64[]
            "st" => []
            "iter" => Int64[]
            "time" => Float64[]
            "output" => IterInfo[]
            ]
        )
    end

    problems = jld2_read("lp-lsq.jld2", "problems")
    if isnothing(problems)
        try
            println("Downloading matrices...")
            download_matrices()
            problems = jld2_read("lp-lsq.jld2", "problems")
        catch
            error("Error while downloading matrices! Delete 'lp-lsq.jld2' and try again.")
        end
    end

    for P in eachrow(problems)
        # desired number of vars per block: ni = n * (nb/100)
        # number of blocks: q = n / ni = 100 / nb
        q = ceil(Int64, 100 / nb)

        print("Instance $(P.name), nb = $(nb)%, p = $(p)")
        if !isempty(
            results[
            (results.run_id .== run_id) .&
            (results.instance .== P.name) .&
            (results.nb .== nb) .&
            (results.p .== p) .&
            (results.blk .== String(nameof(user_blk))) .&
            (results.dec .== String(nameof(user_dec))),:]
        )
            println(" -- already executed")
            continue
        end
        println()

        try
            out, time = solve(
                P.A, P.b;
                q = q,
                user_blk = user_blk,
                user_dec = user_dec,
                usemetis = usemetis,
                p = p,
                verbose = 0
            )

            row = [
                run_id;
                P.name;
                size(P.A);
                nb;
                q;
                p;
                String(nameof(user_blk));
                String(nameof(user_dec));
                out.f;
                out.opt;
                out.status;
                out.iter;
                time;
                out
            ]
            push!(results, (row))

            jldsave(outfile; results)
        catch err
            println("ERROR: $(err)")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # nb = desired number of variables per block (% of variables)

    # Cyclic selection
    println("Cyclic selecion...")
    for nb in [0.5;1.0;5.0;10.0;15.0;20.0]
        run_tests(nb = nb, user_blk = blk_cyclic, user_dec = dec_min)
    end
    for nb in [0.5;1.0;5.0;10.0;15.0;20.0]
        run_tests(nb = nb, user_blk = blk_cyclic, user_dec = dec_max)
    end

    # Results
    println("Compiling results...")
    lplsq_table()
    pp_blk()
    pp_S()
end
