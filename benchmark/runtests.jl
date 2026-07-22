using HSL
using BCD
using LinearAlgebra
using SparseArrays
using MKL
using Pkg
using Random
using DataFrames
using JLD2
using MatrixDepot
using SPGBox
using BenchmarkTools

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

function dec(E, S, iter::IterInfo, par::Param)
    return (iter.iter <= 20) ? max(E,S) : min(E,S)
end
function dec_onlyE(E, S, iter::IterInfo, par::Param)
    return E
end
function dec_onlyS(E, S, iter::IterInfo, par::Param)
    return S
end

function solve(
    A, b, q, user_blk, user_dec, hist, par;
    p = 1.5,
    verbose = 1,
    benchmark = false
)
    @assert length(b) == size(A,1) "Dimensions mismatch"

    n = size(A, 2)

    D = 10^3

    fs = Float64[]
    sigs = Float64[]
    opts = Float64[]

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
    function f(x, s, bs, i, data)
        if (i > 0)
            # compute Ai * s
            @inbounds data.Asi .= data.A[i] * s
            # update A*x - b
            @. data.Axb += data.Asi
        end
        return (1/p) * norm(data.Axb, p)^p
    end

    # partial gradient
    function g!(g, x, bs, i, data)
        @inbounds @views if p == 2.0
            g[bs[i].idx] .= transpose(data.A[i]) * data.Axb
        else
            g[bs[i].idx] .= transpose(data.A[i]) * (abs.(data.Axb).^(p-1) .* sign.(data.Axb))
        end
    end

    # update B_i
    function B(x, bs, i, data)
        @inbounds ni = bs[i].ni
        @inbounds if p == 2.0
            return transpose(data.A[i]) * data.A[i]
        else
            return (p-1)*transpose(data.A[i]) * spdiagm(min.(abs.(data.Axb).^(p-2), D)) * data.A[i]
        end
    end

    function callback(
        x, blocks::Vector{Block}, iter::IterInfo, par::Param
    )
        push!(fs, iter.f)
        push!(sigs, iter.sig)
        push!(opts, iter.opt)
    end

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
    blocks = create_blocks(q, bl_idx)

    time = @elapsed output = bcd(
        blocks, f, g!, B, data_initialize;
        par = par,
        user_blk = user_blk,
        user_dec = user_dec,
        user_callback = hist ? callback : nothing,
        verbose = verbose
    )

    if benchmark
        time = @belapsed begin bcd(
            $blocks, $f, $g!, $B, $data_initialize;
            par = $par,
            user_blk = $user_blk,
            user_dec = $user_dec,
            user_callback = $hist ? $callback : nothing,
            verbose = $verbose
        )
        end samples = 100 seconds = 10.0 evals = 1 gcsample = false gctrial = true
    end

    return output, fs, sigs, opts, time
end

function solve_spg(A, b, par; p = 1.5, verbose = 1, benchmark = false)
    Axb = deepcopy(b)

    # f(x) = 1/p |Ax - b|_p^p
    function f(x)
        @views Axb .= A*x .- b
        return (1/p) * norm(Axb, p)^p
    end

    # gradient
    function g!(g, x)
        g .= transpose(A) * (abs.(Axb).^(p-1) .* sign.(Axb))
    end

    time = @elapsed begin
        x = zeros(size(A,2))
        output = spgbox!(
            f, g!, x,
            iprint = verbose,
            project_x0 = false,
            nitmax = par.maxit,
            nfevalmax = 10^9,
            eps = par.eps
        )
    end

    if benchmark
        time = @belapsed begin
            x = zeros(size($A,2))
            spgbox!(
                $f, $g!, x,
                iprint = $verbose,
                project_x0 = false,
                nitmax = $par.maxit,
                nfevalmax = 10^9,
                eps = $par.eps
            )
        end samples = 100 seconds = 10.0 evals = 1 gcsample = false gctrial = true
    end

    return output, time
end

function run_tests(
    run_id, nb, user_blk, user_dec, hist, par; p = 1.5, benchmark = false
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
            "fs" => Vector{Float64}[]
            "sigs" => Vector{Float64}[]
            "opts" => Vector{Float64}[]
            "alpha" => Float64[]
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

        print("Run id $(run_id), blk $(user_blk), dec $(user_dec), nb = $(nb)% , instance $(P.name)")
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

        if run_id == -1

            # SPG
            try
                out, time = solve_spg(
                    P.A, P.b, par; verbose = 0, benchmark = benchmark
                )

                row = [
                    -1;
                    P.name;
                    size(P.A);
                    100;
                    1;
                    p;
                    "none";
                    "none";
                    out.f;
                    out.gnorm;
                    (out.gnorm <= par.eps) ? 0 : 1;
                    out.nit;
                    time;
                    IterInfo(
                        out.nit,
                        (out.gnorm <= par.eps) ? 0 : 1,
                        out.x,
                        out.f,
                        0.0,
                        out.gnorm,
                        out.nfeval,
                        out.nit,
                        0
                    );
                    [[]];
                    [[]];
                    [[]];
                    0.0
                ]
                push!(results, (row))

                jldsave(outfile; results)
            catch err
                println("ERROR while solving $(P.name) by SPG")
            end

        else

            # BCD
            try
                out, fs, sigs, opts, time = solve(
                    P.A, P.b, q, user_blk, user_dec, hist, par;
                    verbose = 0, benchmark = benchmark
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
                    out;
                    [fs];
                    [sigs];
                    [opts];
                    par.alpha
                ]
                push!(results, (row))

                jldsave(outfile; results)
            catch err
                println("ERROR while solving $(P.name) by BCD")
            end
        end
    end
end

function run_all()
    # nb = desired number of variables per block (% of variables)

    par = default_params()
    par.eps = 1e-3

    # run_id = 1 to 7: calibrate alpha
    par.maxit = max(5000, 100 * 10)
    par.maxfnoimpr = ceil(Int64, par.maxit/5)
    for (run_id, alpha) in enumerate([1e-1; 1e-2; 1e-3; 1e-4; 1e-5; 1e-6; 1e-7])
        par.alpha = alpha
        run_tests(run_id, 10.0, blk_cyclic, dec_min, true, par)
        run_tests(run_id, 10.0, blk_cyclic, dec_max, true, par)
        run_tests(run_id, 10.0, blk_cyclic, dec_onlyE, true, par)
        #run_tests(run_id, 10.0, blk_cyclic, dec_onlyS, true, par)
    end

    # run_id = 0: table, varying nb
    par.alpha = 1e-6
    for nb in [0.5;1.0;5.0;10.0;15.0;20.0]
        par.maxit = max(5000, 100 * ceil(Int64, 100 / nb))
        par.maxfnoimpr = ceil(Int64, par.maxit/5)
        run_tests(0, nb, blk_cyclic, dec_min, false, par, benchmark = true)
        #run_tests(0, nb, blk_cyclic, dec_max, false, par)
        #run_tests(0, nb, blk_cyclic, dec_onlyE, false, par)
        #run_tests(0, nb, blk_cyclic, dec_onlyS, false, par)
    end

    # run_id = -1: SPG
    par.maxit = 5000
    run_tests(-1, 100, blk_cyclic, dec_min, false, par, benchmark = true)

    # Results
    println("Compiling results...")
    statistics()
    lplsq_table()
    pp()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
