#

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators
using Random
# using BregmanBC
using Flux
import ProximalOperators: gradient
export solution

include("SGD_prox.jl")
include("SGD_prox_DNN.jl")
include("GD_prox_DNN.jl")


struct SGD{R<:Real}
    γ::Maybe{R}
    maxit::Int
    verbose::Bool
    freq::Int
    plus::Bool # true for diminishing stepsize
    DNN::Bool
    η0::Maybe{R}
    η_tilde::Maybe{R}
    GD::Bool
    function SGD{R}(;
        γ::Maybe{R} = nothing,
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 1000,
        plus::Bool = false,
        DNN::Bool = false,
        η0::Maybe{R} = 0.1,
        η_tilde::Maybe{R} = 0.5,
        GD::Bool = false
    ) where {R}
        @assert γ === nothing || γ > 0
        @assert maxit > 0
        @assert freq > 0
        @assert η0 > 0
        @assert η_tilde > 0
        new(γ, maxit, verbose, freq, plus, DNN, η0, η_tilde, GD)
    end
end

function (solver::SGD{R})(
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    μ = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}

    stop(state::SVRG_basic_state) = false
    disp(it, state) = @printf "%5d | %.3e  \n" it state.γ

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))

    maxit = solver.maxit

    # dispatching the structure
    iter = SGD_prox_iterable(F, g, x0, N, L, μ, solver.γ)
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
    SGD([γ, maxit, verbose, freq])

Instantiate the SGD algorithm  for solving (strongly) convex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x) 

If `solver = SGD(args...)`, then the above problem is solved with

	solver(x0, [F, g, N, L, μ])

where F is an array containing f_i's, x0 is the initial point, and L, μ are arrays of 
smoothness and strong convexity moduli of f_i's; they are optional when γ is provided.  

Optional keyword arguments are:
* `γ`: stepsize  
* `L`: an array of smoothness moduli of f_i's 
* `μ`: (if strongly convex) an array of strong convexity moduli of f_i's 
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.

"""

SGD(::Type{R}; kwargs...) where {R} = SGD{R}(; kwargs...)
SGD(; kwargs...) = SGD(Float64; kwargs...)


"""
If `solver = SVRG(args...)`, then 

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
    solver::SGD{R},
    x0::Union{AbstractArray{C},Tp};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    μ = nothing,
    N = N,
    data = nothing,
    DNN_config::Tdnn
) where {R,C<:RealOrComplex{R},Tp,Tdnn}
    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    if solver.DNN
        L = 1.0
        if solver.GD
            println("GD prox version - DNN")
            iter = GD_prox_DNN_iterable(F, g, x0, N, L, μ, solver.γ, solver.plus, solver.η0, solver.η_tilde, data, DNN_config)
        else
            println("SGD prox version - DNN")
            iter = SGD_prox_DNN_iterable(F, g, x0, N, L, μ, solver.γ, solver.plus, solver.η0, solver.η_tilde, data, DNN_config)
        end
    else
        println("SGD prox version")
        iter = SGD_prox_iterable(F, g, x0, N, L, μ, solver.γ, solver.plus, solver.η0, solver.η_tilde)
    end
    return iter
end
