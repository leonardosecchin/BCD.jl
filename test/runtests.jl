using BCD
using Test
using LinearAlgebra
using SparseArrays

@testset "Simple test" begin
    struct DATA
        A::Vector{SparseMatrixCSC{Float64, Int64}}
        Axb::Vector{Float64}
        Asi::Vector{Float64}
        B::Vector{SparseMatrixCSC{Float64, Int64}}
        b::Vector{Float64}
    end

    A = Symmetric([1.0 0.5 0.1; 0.0 2.0 0.0; 0.0 0.0 3.0])
    b = A * ones(3)

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
        return 0.5 * norm(data.Axb, 2)^2
    end

    # partial gradient
    function g!(g, bs, i, data)
        @inbounds @views g[bs[i].idx] .= transpose(data.A[i]) * data.Axb
    end

    # update B_i
    function B(bs, i, data)
        return @inbounds transpose(data.A[i]) * data.A[i]
    end

    blocks = create_blocks(3, [1;2;3])
    output = bcd(blocks, f, g!, B, data_initialize; verbose = 0)
    @test output.status == 0
end
