"""
Structure for iteration information.

## Fields

- `iter    :: Int64`: number of iterations
- `status  :: Int64`: exit flag (type `?bcd` for details)
- `x       :: Vector{Float64}`: iterate
- `f       :: Float64`: objective value
- `sig     :: Float64`: penalization parameter
- `opt     :: Float64`: maximum supnorm over all partial gradients
- `nf      :: Int64`: number of evaluations of the objective function
- `ng      :: Int64`: number of evaluations of the partial gradients
- `nB      :: Int64`: number of evaluations of the approximate Hessian
"""
mutable struct IterInfo
    iter    ::Int64
    status  ::Int64
    x       ::Vector{Float64}
    f       ::Float64
    sig     ::Float64
    opt     ::Float64
    nf      ::Int64
    ng      ::Int64
    nB      ::Int64
end

"""
Structure for parameters.

## Fields

- `eps       ::Float64`: optimality tolerance
- `alpha     ::Float64`: line search parameter
- `theta     ::Float64`: inexactness level allowed when computing directions
- `maxit     ::Int64`: maximum number of iterations allowed
- `sig0      ::Float64`: initial penalization
- `fest      ::Float64`: stop if objective function value ≤ `fest`
- `maxsig    ::Float64`: maximum penalization allowes
- `maxfnoimpr::Int64`: maximum number of consecutive iterations without
  improvement allowed
"""
mutable struct Param
    eps       ::Float64
    alpha     ::Float64
    theta     ::Float64
    maxit     ::Int64
    sig0      ::Float64
    fest      ::Float64
    maxsig    ::Float64
    maxfnoimpr::Int64
end

"""
Returns a `Param` structure with default values.
"""
function default_params()
    return Param(1e-4, 1e-4, 1.0, 10000, 1.0, -Inf, 1e+20, 0)
end

# convert a Vector{Int64} into a UnitRange if possible
function consec_range(v::Vector{Int64})
    if isempty(v)
        return v
    else
        sort!(v)
        @inbounds if v[end] - v[1] + 1 == length(v)
            return v[1]:v[end]
        else
            return v
        end
    end
end
