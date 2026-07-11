"""
Block Coordinate Descent method to minimize Hölder continuous functions.
"""
module BCD

using HSL
using LinearAlgebra
using SparseArrays
using Printf
using Random

include("basic.jl")
include("blocks.jl")
include("descent.jl")

# structures
export Block, IterInfo, Param
# auxiliary functions
export default_params
export dec_min, dec_max
export blk_cyclic, blk_random, blk_max
# main functions
export create_blocks, bcd

"""
    output = bcd(blocks, f, g!, B, data_init; [OPTIONS])

This is the Block Coordinate Descent method described in

`Amaral, Andreani, Secchin, Silva. Flexible block coordinate descent methods
for unconstrained optimization under Hölder continuity. 2026`

The user must provide the vector `blocks` of `Block` strucure, the functions
`f` and `g` to evaluate the objective function and its partial gradient, the
function `B` that returns the sparse partial Hessian approximation, and the
function `data_init` that initializes problem-specific structures. Their headers
should be:

`function f(s, blocks::Vector{Block}, bid, data)`\\
`function g!(g, blocks::Vector{Block}, bid, data)`\\
`function B(blocks::Vector{Block}, bid, data)`\\
`function data_initialize(x, blocks::Vector{Block})`

where `x` is the full vector of variables, `bid` is the block index and `data`
is a `struct` with all necessary stuff for the problem, defined by the user.
In the function `g!`, `g` is the full gradient and the partial gradient, whose
entries associated with block index `bid` receive the corresponding partial
gradient at the current point. Function `f` should return a `Float64`, function
`B!` should return the matrix `B` in sparse format, and `data_initialize` the
structure `data` initialized at `x`. Only lower triangle of `B` should be
provided. Note that fields in `data` are modified during the method, so it must
be `mutable` if it contains numeric fields.

It is important to note that `f` receives the partial direction `s`, that is,
the vector of components of `x+ - x` in the block `i`, where `x+` is the point
`x` after a step w.r.t. the block `i`. The functions `g!` and `B` do not receive
`x`; if needed, the current point must be stored as a property of `data`.

`output` is a `IterInfo` structure that contains information about the
resolution process. In particular, `output.x` is the final iterate and
`output.status` is the exit status. For a complete list os properties, type
`?IterInfo`.

## Exit status (`output.status`)

- `0`:  stationary point found
- `1`:  acceptable point found
- `2`:  stop with large penalization
- `3`:  maximum number of iterations reached
- `4`:  lack of progress
- `91`: x0 is inconsistent
- `92`: error while computing B
- `93`: invalid B
- `94`: error while factorizing the Hessian approximation (HSL MA57)
- `95`: error while solving the subproblem

## Options

- `par`: parameter structure `Param`
- `x0`: initial guess (default `empty`)
- `user_f`: descent criterion (default `dec_min`)
- `user_blk`: function for selecting the block (default `blk_random`)
- `user_callback`: user-defined callback function (default `nothing`)
- `seed`: random seed (<0 for any; default `-1`)
- `verbose`: output level (`1` = standard, `0` = none; default `0`)

`user_dec` can be `dec_min` or `dec_max`. These correspond to the descent
criteria `min{E,S}` and `max{E,S}` in the reference paper. You can defined our
own function, which should return a number and should have the header

`user_dec(E, S, iter::IterInfo, par::Param)`.

Type `?IterInfo` and `?Param` to view the available properties.

The function `user_blk` should return the index of the next block and should
have the header

`user_next(blocks, curr_id, elegible, opts)`,

where `blocks` is the vector of blocks, `curr_id` receives the ID of the current
block, `elegible` is a `BitVector` indicating which blocks are elegible for
selection, and `opts` receives a vector of the supnorm of each partial gradient.
`blk_cyclic` (ascending cyclic rule) and `blk_random` (random selecting) are
pre-implemented and can be passed. If you provide your own function, ensure that
all blocks are visited during a cycle.

A callback function can be defined. It is executed at the beginning of each
iteration, and can be used to set additional stopping criteria. The header is

`user_callback(x, blocks::Vector{Block}, iter::IterInfo, par::Param)`

## Modifying parameters

You can initialize the `Param` structure with default values executing
`par = default_params()` and then modify its values. For example, to set
maximum number of iterations to 1,000, simply do `par.maxit = 1000`.

## Example

For a complete example, see the `example` folder in the repository.
"""
function bcd(
    blocks     ::Vector{Block},               # vector of blocks
    f          ::Function,                    # objective function
    g!         ::Function,                    # gradient of f
    B          ::Function,                    # B update rule
    data_init  ::Function;                    # initialize data function
    par        ::Param    = default_params(), # parameters structure
    x0                    = [],               # initial guess
    user_dec   ::Function = dec_min,          # descent criteria for f
    user_blk   ::Function = blk_random,       # rule for chosing the next block
    user_callback         = nothing,          # callback
    seed       ::Int64    = -1,               # random seed
    verbose               = 1                 # output level
)

    @assert length(blocks) > 0 throw(ArgumentError("At least one block must be defined"))

    # seed for any randomization
    if seed >= 0
        Random.seed!(seed)
    end

    user_blk(blocks, -1, falses(length(blocks)), fill(Inf, length(blocks)))

    # total and maximum number of variables in blocks
    n = 0
    maxni = 0
    for i in eachindex(blocks)
        n += blocks[i].ni
        maxni = max(maxni, blocks[i].ni)
    end

    # initialize iteration information structure
    iter = IterInfo(0, 9, zeros(Float64, n), Inf, par.sig0, Inf, 0, 0, 0)

    # initial point
    if !isempty(x0)
        try
            iter.x .= x0
        catch
            if verbose > 0
                @error "Invalid initial point"
            end
            iter.status = 91
            return iter
        end
    end

    # intialize data
    data = data_init(iter.x, blocks)

    # working vectors
    xtrial = deepcopy(iter.x)

    g  = similar(iter.x)
    xi = Vector{Float64}(undef, maxni)
    si = similar(xi)
    prev_si = similar(xi)

    maxfnoimpr = (par.maxfnoimpr > 0) ? par.maxfnoimpr : max(10, 2 * length(blocks))

    opts  = fill(Inf, length(blocks))
    lastf = fill(Inf, maxfnoimpr)

    # for MA57
    ma57work = similar(xi)

    # indexes of eligible blocks
    eligible_blks = trues(length(blocks))

    # block index
    bid = 0

    # evaluate f at the initial point
    # data is up to date regarding the current point, so we pass block id 0
    iter.f = f([], blocks, 0, data)
    iter.nf += 1

    @inbounds lastf[1] = iter.f

    bid = 0
    prohibited_blk = false

    E = 0.0
    S = 0.0

    # print initial banner
    if verbose > 0
        printbanner(blocks, par)
    end

    # MAIN LOOP
    @inbounds while (true)

        # test whether x is an acceptable solution
        if iter.f <= par.fest
            printiter(iter, bid, verbose, true)
            if verbose > 0
                println("\nEXIT STATUS: An acceptable solution was found")
            end
            iter.status = 1
            return iter
        end

        # test whether maximum number of iterations was reached
        if iter.iter >= par.maxit
            printiter(iter, bid, verbose, true)
            if verbose > 0
                println("\nEXIT STATUS: Maximum number of iterations reached")
            end
            iter.status = 3
            return iter
        end

        # user callback
        if !isnothing(user_callback)
            user_callback(iter.x, blocks, iter, par)
        end

        # test whether f did not improved
        if maximum(lastf) <= iter.f + 1e-8 * max(1.0, iter.f)
            printiter(iter, bid, verbose, true)
            if verbose > 0
                println("\nEXIT STATUS: Lack of progress")
            end
            iter.status = 4
            return iter
        end

        lastf[mod(iter.iter, maxfnoimpr) + 1] = iter.f

        # turn all blocks elegible
        eligible_blks .= true

        # is current blk prohibited?
        if prohibited_blk
            eligible_blks[bid] = false
        end

        # stationarity test and block choice
        while (true)
            # test whether iter.x is stationary (no block is elegible)
            if !any(eligible_blks)
                printiter(iter, bid, verbose, true)
                if verbose > 0
                    println("\nEXIT STATUS: An approximate stationary point was found")
                end
                iter.status = 0
                return iter
            end

            # choose a block
            bid = user_blk(blocks, bid, eligible_blks, opts)

            # compute partial grad f
            g!(g, blocks, bid, data)
            iter.ng += 1

            @views opts[bid] = norm(g[blocks[bid].idx], Inf)
            iter.opt = maximum(opts)

            if opts[bid] <= par.eps
                # remove bid from C
                eligible_blks[bid] = false
            else
                # resize working vectors to handle ma57 objects correctly
                resize!(ma57work, blocks[bid].ni)
                resize!(xi, blocks[bid].ni)
                resize!(si, blocks[bid].ni)
                resize!(prev_si, blocks[bid].ni)

                # copy partial iter.x to xi and proceed
                @views xi .= iter.x[blocks[bid].idx]
                break
            end
        end

        # print iteration information
        printiter(iter, bid, verbose, false)

        # reset sigma
        iter.sig = par.sig0
        prohibited_blk = false

        # Instantiate objects associated with factorization
        MA = nothing

        sisupn = Inf

        # compute B
        # only the lower triangle of B must be given
        Bi = B(blocks, bid, data)
        iter.nB += 1

        if isempty(Bi)
            printiter(iter, bid, verbose, true)
            if verbose > 0
                @error "Error computing B"
            end
            iter.status = 92
            return iter
        end

        # subproblem, line search w.r.t. the current block
        while (true)

            # if sig is large, pass to the next block
            if iter.sig > par.maxsig
                prohibited_blk = true
                break
            end

            prev_si .= si

            # QUADRATIC REGULARIZATION, descent direction si

            # BsigI = B + 2sig*I
            ni = blocks[bid].ni
            BsigI = Bi + 2.0 * iter.sig * spdiagm(ni, ni, ones(ni))

            if LIBHSL_isfunctional()
                # HSL is working
                MA = nothing
                try
                    # create Ma57 object
                    MA = Ma57(BsigI)
                catch
                    printiter(iter, bid, verbose, true)
                    if verbose > 0
                        @error "ma57 object could not be initalized. Is B valid?"
                    end
                    iter.status = 93
                    return iter
                end

                # factorize
                ma57_factorize!(MA)

                if MA.info.info[1] < 0
                    printiter(iter, bid, verbose, true)
                    if verbose > 0
                        @error "ma57 factorization failed"
                    end
                    iter.status = 94
                    return iter
                end

                if (MA.info.num_negative_eigs > 0) && (verbose > 1)
                    @warn "B + 2sig*I is not positive semidefinite"
                end

                # solve BsigI*s = -g
                @views si .= -g[blocks[bid].idx]   # rhs
                ma57_solve!(MA, si, ma57work, job=:A)

                if MA.info.info[1] < 0
                    printiter(iter, bid, verbose, true)
                    if verbose > 0
                        @error "Subproblem resolution failed (MA57)"
                    end
                    iter.status = 95
                    return iter
                end
            else
                # compute si by the native Julia linear system solver
                # positiveness is not checked
                try
                    @views si .= BsigI \ (-g[blocks[bid].idx])
                catch
                    printiter(iter, bid, verbose, true)
                    if verbose > 0
                        @error "Subproblem resolution failed (Julia solver)"
                    end
                    iter.status = 95
                    return iter
                end
            end

            E = (par.alpha / (16.0 * iter.sig)) * par.eps^2
            S = par.alpha * dot(si, si)

            sisupn = norm(si, Inf)

            # if the previous and new directions are "equal", breaks
            norm_s = max(1.0, max(norm(prev_si,Inf), sisupn))
            if norm(prev_si .- si, Inf) <= 1e-14 * norm_s
                break
            end

            # xtrial = x + s
            @views @. xtrial[blocks[bid].idx] = xi + si

            # descent condition
            ftrial = f(si, blocks, bid, data)
            iter.nf += 1

            dec = user_dec(E, S, iter, par)
            norm_f = max(1.0, max(abs(ftrial), abs(iter.f)))

            # accepts the step if "normalized f" decreases or direction is too small
            if (ftrial/norm_f <= iter.f/norm_f - dec + eps(norm_f)) ||
               (sisupn <= 1e-12 * norm(xi, Inf))
                # success, update iter.x
                @views iter.x[blocks[bid].idx] .= xtrial[blocks[bid].idx]

                # update f for the next iteration
                iter.f = ftrial

                # next iteration
                break
            else
                # increase sig and try again
                iter.sig = max(1.0, 10.0 * iter.sig)
            end
        end

        iter.iter += 1
    end
end

function printiter(iter::IterInfo, bid, verbose, final)
    if verbose > 0
        if (mod(iter.iter,1000) == 0) && !final
            println("\n    iter |         f |   max opt |  curr blk |         σ")
        end
        if (mod(iter.iter,50) == 0) || final
            @printf(
                " %7d%s| %9.2e | %9.2e | %9d | %9.2e\n",
                iter.iter, final ? "*" : " ", iter.f, iter.opt, bid, iter.sig
            )
        end
    end
end

function printbanner(blocks::Vector{Block}, par::Param)
    minblksize = maxblksize = blocks[1].ni
    n = 0
    for k in eachindex(blocks)
        minblksize = min(minblksize, blocks[k].ni)
        maxblksize = max(maxblksize, blocks[k].ni)
        n += blocks[k].ni
    end
    println("\nThis is the Block Coordinate Descent method described in")
    println()
    println(" Amaral, Andreani, Secchin, Silva. Flexible block")
    println(" coordinate descent methods for unconstrained")
    println(" optimization under Hölder continuity. 2026")
    println()
    if LIBHSL_isfunctional()
    println("HSL MA57 is present.\n")
    end
    println('='^18," Problem statistics ",'='^18)
    @printf("Total number of variables                      %9d\n", n)
    @printf("Number of blocks                               %9d\n", length(blocks))
    @printf("Smallest block size                            %9d\n", minblksize)
    @printf("Largest block size                             %9d\n", maxblksize)
    println()
    println('='^22," Parameters ",'='^22)
    @printf("Optimality tolerance                           %9.2e\n", par.eps)
    @printf("Maximum number of iterations                   %9d\n"  , par.maxit)
    @printf("Acceptable objetive value                      %9.2e\n", par.fest)
    @printf("Initial penalization (σ)                       %9.2e\n", par.sig0)
    @printf("Maximum penalization allowed                   %9.2e\n", par.maxsig)
    @printf("α                                              %9.2e\n", par.alpha)
    @printf("θ                                              %9.2e\n", par.theta)
    @printf("Max number of iterations without improvement   %9d\n"  ,
        (par.maxfnoimpr > 0) ? par.maxfnoimpr : 2 * length(blocks))
    println('='^56)
end

end
