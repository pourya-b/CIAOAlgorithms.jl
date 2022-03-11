struct SVRG_prox_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    μ::Maybe{Union{Array{R},R}}  # convexity moduli of the gradients
    γ::Maybe{R}             # stepsize 
    m::Maybe{Int}           # number of inner loop updates
end

mutable struct SVRG_prox_state{R<:Real,Tx}
    γ::R                    # stepsize 
    m::Int                  # number of inner loop updates
    av::Tx                  # the running average
    z::Tx
    z_full::Tx              # the outer loop argument
    w::Tx                   # the inner loop variable
    ind::Array{Int}         # running idx set 
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
end

function SVRG_prox_state(γ::R, m, av::Tx, z::Tx, z_full::Tx, w::Tx, ind) where {R,Tx}
    return SVRG_prox_state{R,Tx}(γ, m, av, z, z_full, w, ind, copy(av), copy(av))
end

function Base.iterate(iter::SVRG_prox_iterable{R}) where {R}
    N = iter.N
    ind = collect(1:N)
    m = iter.m === nothing ? m = N : m = iter.m
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "smoothness parameter absent"
            return nothing
        elseif iter.μ === nothing 
            L_M = maximum(iter.L)
            γ = 1 / (3 * N * L_M) # according to thm 1 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization", for batch size = 1
        else
            L_M = maximum(iter.L)
            μ_M = maximum(iter.μ)
            γ = 0.99 / (4 * L_M) # according to thm 3.1 of "A PROXIMAL STOCHASTIC GRADIENT METHOD WITH PROGRESSIVE VARIANCE REDUCTION" 
            rho = (1 + 4 * L_M * γ^2 * μ_M * (m + 1)) / (μ_M * γ * m * (1 - 4L_M * γ))
            if rho >= 1
                @warn "convergence condition violated...provide a stepsize!"
                return nothing
            end
        end
    else
        γ = iter.γ # provided γ
    end
    # initializing the vectors 
    av = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        ∇f ./= N
        av .+= ∇f
    end
    z_full = copy(iter.x0)
    z = zero(av)
    w = copy(iter.x0)
    state = SVRG_prox_state(γ, m, av, z, z_full, w, ind)
    return state, state
end

function Base.iterate(iter::SVRG_prox_iterable{R}, state::SVRG_prox_state{R}) where {R}
    # The inner cycle
    for i in rand(state.ind, state.m)  # Uniformly randomly pick one index \in [N] (with replacement) in each iteration
        gradient!(state.temp, iter.F[i], state.z_full)
        gradient!(state.∇f_temp, iter.F[i], state.w)
        state.temp .-= state.∇f_temp
        state.temp .-= state.av
        state.temp .*= state.γ
        state.temp .+= state.w
        CIAOAlgorithms.prox!(state.w, iter.g, state.temp, state.γ)
        state.z .+= state.w   # keeping track of the sum of w's
    end

    # full update 	
    state.z_full .= state.w # state.z ./ state.m
    state.z = zero(state.z)  
    state.av .= state.z
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.F[i], state.z_full)
        state.∇f_temp ./= iter.N
        state.av .+= state.∇f_temp
    end

    return state, state
end


solution(state::SVRG_prox_state) = state.z_full
