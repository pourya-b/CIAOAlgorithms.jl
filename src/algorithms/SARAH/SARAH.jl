# Nguyen, Lam M., et al. "SARAH: A novel method for machine learning problems using stochastic recursive gradient." 
# International Conference on Machine Learning. PMLR, 2017.

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators
using Random
# using Flux # for DNN training

export solution

include("SARAH_basic.jl") 
include("SARAH_prox.jl")
include("SARAH_prox_DNN.jl")

struct SARAH{R<:Real}
    γ::Maybe{R}
    maxit::Int
    verbose::Bool
    freq::Int
    m::Maybe{Int}
    ꞵ::Maybe{R}    # scalar momentum 
    DNN::Bool
    function SARAH{R}(;
        γ::Maybe{R} = nothing,
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 1000,
        m::Maybe{Int} = nothing, # number of inner loop updates
        ꞵ::Maybe{R} = nothing,
        DNN::Bool = false,
    ) where {R}
        @assert γ === nothing || γ > 0
        @assert maxit > 0
        @assert freq > 0
        new(γ, maxit, verbose, freq, m, ꞵ, DNN)
    end
end

function (solver::SARAH{R})(
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    μ = nothing,
    N = nothing,
) where {R,C<:RealOrComplex{R}}

    stop(state::SARAH_basic_state) = false
    disp(it, state) = @printf "%5d | %.3e  \n" it state.γ

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    m = solver.m === nothing ? m = N : m = solver.m

    maxit = solver.maxit

    # dispatching the structure
    if g == ProximalOperators.Zero()
        iter = SARAH_basic_iterable(F, x0, N, L, μ, solver.γ, m)
    else
        if solver.DNN
            iter = SARAH_prox_DNN_iterable(F, g, x0, N, L, solver.γ, m, solver.ꞵ, data, DNN_config)
        else
            iter = SARAH_prox_iterable(F, g, x0, N, L, μ, solver.γ, m, solver.ꞵ)
        end
    end

    iter = take(halt(iter, stop), maxit)
    iter = enumerate(iter)
    num_iters, state_final = nothing, nothing
    for (it_, state_) in iter  # unrolling the iterator 
        # see https://docs.julialang.org/en/v1/manual/interfaces/index.html
        if solver.verbose && mod(it_, solver.freq) == 0
            disp(it_, state_)
        end
        num_iters, state_final = it_, state_
    end
    if solver.verbose && mod(num_iters, solver.freq) !== 0
        disp(num_iters, state_final)
    end # for the final iteration
    return solution(state_final), num_iters
end


"""
    SARAH([γ, maxit, verbose, freq, m])

Instantiate the SARAH algorithm  for solving (strongly) convex and nonconvex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x)

If `solver = SARAH(args...)`, then the above problem is solved with

	solver(x0, [F, g, N, L, μ])

where F is an array containing f_i's, x0 is the initial point, and L, μ are arrays of 
smoothness and strong convexity moduli of f_i's; they are optional when γ is provided.  

Optional keyword arguments are:
* `γ`: stepsize  
* `L`: an array of smoothness moduli of f_i's. Not considered if γ is provided.
* `μ`: an array of strong convexity moduli of f_i's. Not considered if γ is provided.
    - if provided, the problem is considered as strongly convex, 
    - if it is zero, convex problem is considred, 
    - if nothing is provided, a nonconvex problem is considred. 
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.

"""

SARAH(::Type{R}; kwargs...) where {R} = SARAH{R}(; kwargs...)
SARAH(; kwargs...) = SARAH(Float64; kwargs...)


"""
If `solver = SARAH(args...)`, then 
    itr = iterator(solver, x0, [F, g, N, L, μ])
is an iterable object. Note that [maxit, verbose, freq] fields of the solver are ignored here. 

The solution at any given state can be obtained using solution(state), e.g., 
for state in Iterators.take(itr, maxit)
    # do something using solution(state)
end

See https://docs.julialang.org/en/v1/manual/interfaces/index.html 
and https://docs.julialang.org/en/v1/base/iterators/ for a list of iteration utilities
"""

function iterator(
    solver::SARAH{R},
    x0::Union{AbstractArray{C},Tp};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    μ = nothing,
    N = nothing,
    data = nothing, # training data for DNN training
    DNN_config::Maybe{Tdnn} = nothing
) where {R,C<:RealOrComplex{R},Tdnn,Tp}
    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    m = solver.m === nothing ? m = N : m = solver.m

    # dispatching the structure
    if g == ProximalOperators.Zero()
        iter = SARAH_basic_iterable(F, x0, N, L, μ, solver.γ, m)
    else
        if solver.DNN
            iter = SARAH_prox_DNN_iterable(F, g, x0, N, L, solver.γ, m, solver.ꞵ, data, DNN_config)
        else
            iter = SARAH_prox_iterable(F, g, x0, N, L, μ, solver.γ, m, solver.ꞵ)
        end
    end

    return iter
end
