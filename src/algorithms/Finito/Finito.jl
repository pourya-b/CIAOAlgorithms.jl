# Latafat, Themelis, Patrinos, "Block-coordinate and incremental aggregated
# proximal gradient methods for nonsmooth nonconvex problems."
# arXiv:1906.10053 (2019).
#
# Latafat. "Distributed proximal algorithms for large-scale structured optimization"
# PhD thesis, KU Leuven, 7 2020.
#
# Mairal, "Incremental majorization-minimization optimization with application to
# large-scale machine learning."
# SIAM Journal on Optimization 25, 2 (2015), 829–855.
#
# Defazio, Domke, "Finito: A faster, permutable incremental gradient method
# for big data problems."
# In International Conference on Machine Learning (2014), pp. 1125-1133.
#

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using ProximalAlgorithms: LBFGS, update!, mul!
using Base.Iterators # to use take and enumerate functions
using Random
using StatsBase: sample
using Flux           # for DNN training

import ProximalOperators: prox!, prox, gradient, gradient!
export solution, epoch_count

abstract type CIAO_iterable end 

include("Finito_basic.jl")                  # Finito/MISO
include("Finito_LFinito.jl")                # SPIRAL-no-ls
include("Finito_LFinito_lbfgs.jl")          # SPIRAL
include("Finito_LFinito_lbfgs_adaptive.jl") # adaSPIRAL
include("Finito_LFinito_lbfgs_DNN.jl")      # SPIRAL for DNN

# include("Finito_adaptive.jl")
# include("Finito_DLFinito.jl")
# include("Finito_DLFinito_lbfgs.jl")
# include("Finito_LFinito_lbfgs_adaptive_DNN.jl")

struct Finito{R<:Real} # solver
    γ::Maybe{Union{Array{R},R}}     #stepsize
    sweeping::Int8                  # sweeping strategy in the inner loop, # 1:cyclical, 2:shuffled
    LFinito::Bool                   # if true, no-ls version
    lbfgs::Bool
    memory::Int                     # lbfgs memory size
    η::R                            # ls parameter for γ
    β::R                            # ls parameter for τ (interpolation between the quasi-Newton direction and the nominal step)
    adaptive::Bool
    DeepLFinito::Tuple{Bool,Int,Int}
    minibatch::Tuple{Bool,Int}
    maxit::Int
    verbose::Bool
    freq::Int
    α::R                            # in (0, 1) for stepsize
    tol::R                          # γ backtracking stopping criterion
    ls_tol::R                       # tolerance in ls
    D::Maybe{R}                     # a large number to normalize lbfgs direction if provided
    DNN_training::Bool
    function Finito{R}(;
        γ::Maybe{Union{Array{R},R}} = nothing,
        sweeping = 1,
        LFinito::Bool = false,
        lbfgs::Bool = false,
        memory::Int = 6, 
        η::R = 0.8,
        β::R = 1/50,
        adaptive::Bool = false,
        DeepLFinito::Tuple{Bool,Int, Int} = (false, 3, 3),
        minibatch::Tuple{Bool,Int} = (false, 1),
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 10000,
        α::R = R(0.999), 
        tol::R = R(1e-9),
        ls_tol::R = R(1e-6),        # pretty large to avoid numerical instability!
        D::Maybe{R} = nothing,
        DNN_training::Bool = false,
    ) where {R}
        @assert γ === nothing || minimum(γ) > 0
        @assert maxit > 0
        @assert memory >= 0
        @assert tol > 0
        @assert freq > 0
        new(γ, sweeping, LFinito, lbfgs, memory, η, β, adaptive, DeepLFinito, minibatch, maxit, verbose, freq, α, tol, ls_tol, D, DNN_training)
    end
end

function (solver::Finito{R})( 
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}

    stop(state) = false # the stopping function for halt function
    disp(it, state) = @printf "%5d | %.3e  \n" it state.hat_γ

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    if solver.DNN_training # SPIRAL for DNN
        iter = FINITO_lbfgs_iterable_DNN(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.β,
            solver.α,
            LBFGS(w0, solver.memory),
            data,
            DNN_config,
        )
    elseif solver.LFinito # no-ls
        iter = FINITO_LFinito_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    elseif solver.lbfgs
        if solver.adaptive # adaSPIRAL
            iter = FINITO_lbfgs_adaptive_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.η,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.tol,
                solver.ls_tol,
                solver.D,
            )
        else #SPIRAL
            iter = FINITO_lbfgs_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.ls_tol,
                solver.D,
            )
        end
    else # Finito/MISO
        iter = FINITO_basic_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    end

    iter = halt(iter, stop)
    iter = take(iter, solver.maxit)
    iter = enumerate(iter)

    num_iters, state_final = nothing, nothing
    for (it_, state_) in iter  # unrolling the iterator (acts as tee and loop functions in the tutorial)
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
    Finito([γ, sweeping, LFinito, adaptive, minibatch, maxit, verbose, freq, tol, ls_tol, DNN_training])

Instantiate the Finito algorithm for solving fully nonconvex optimization problems of the form

    minimize 1/N sum_{i=1}^N f_i(x) + g(x)

where `f_i` are smooth and `g` is possibly nonsmooth, all of which may be nonconvex.

If `solver = Finito(args...)`, then the above problem is solved with

	solver(x0, [F, g, N, L])

where F is an array containing f_i's, x0 is the initial point, and L is an array of
smoothness moduli of f_i's; it is optional in the adaptive mode or if γ is provided.

Optional keyword arguments are:
* `γ`: an array of N stepsizes for each coordinate
* `sweeping::Int` 1 for cyclic (default), 2 for shuffled
* `LFinito::Bool` low memory variant of the Finito/MISO algorithm
* `adaptive::Bool` to activate adaptive smoothness parameter computation
* `minibatch::(Bool,Int)` to use batchs of a given size
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10000`), frequency of verbosity.
* `α::R` parameter where γ_i = αN/L_i
* `tol::Real` (default: `1e-8`), absolute tolerance for the adaptive case
* `ls_tol::Real` (default: `eps(R)`), absolute tolerance for the linesearches
* `DNN_training::Bool` (default: `false`), for DNN training version
"""

Finito(::Type{R}; kwargs...) where {R} = Finito{R}(; kwargs...) 
Finito(; kwargs...) = Finito(Float64; kwargs...)


"""
If `solver = Finito(args...)`, then

    itr = iterator(solver, x0, [F, g, N, L])

is an iterable object. Note that [maxit, verbose, freq] fields of the solver are ignored here.

The solution at any given state can be obtained using solution(state), e.g.,
for state in Iterators.take(itr, maxit)
    # do something using solution(state)
end

See https://docs.julialang.org/en/v1/manual/interfaces/index.html
and https://docs.julialang.org/en/v1/base/iterators/ for a list of iteration utilities
"""


function iterator( 
    solver::Finito{R},
    x0::Union{AbstractArray{C},Tp};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
    data = nothing, # training data for DNN training
    DNN_config::Maybe{Tdnn} = nothing
) where {R,C<:RealOrComplex{R},Tp,Tdnn}
    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    DNN_config == nothing ? w0 = copy(x0) : w0 = DNN_config()
    if solver.DNN_training # SPIRAL for DNN
        iter = FINITO_lbfgs_iterable_DNN(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.β,
            solver.α,
            LBFGS(w0, solver.memory),
            data,
            DNN_config,
        )
    elseif solver.LFinito # SPIRAL-no-ls
        iter = FINITO_LFinito_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    elseif solver.lbfgs
        if solver.adaptive # adaSPIRAL
            iter = FINITO_lbfgs_adaptive_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.η,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.tol,
                solver.ls_tol,
                solver.D,
            )
        else #SPIRAL
            iter = FINITO_lbfgs_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.ls_tol,
                solver.D,
            )
        end
    else # Finito/MISO
        iter = FINITO_basic_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    end
    return iter
end

## TO DO:
# more stable adaptive mode!