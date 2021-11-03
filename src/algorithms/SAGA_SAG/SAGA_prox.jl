struct SAGA_prox_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    γ::Maybe{R}             # stepsize 
    SAG::Bool               # to activate SAG version
end

mutable struct SAGA_prox_state{R<:Real,Tx}
    a::Array{Tx}            # table of alphas
    γ::R                    # stepsize 
    av::Tx                  # the running average
    z::Tx                   # the latest iteration
    # some extra placeholders 
    ind::Int                # running idx set 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx                # placeholder for gradients 
    w::Tx                   # input of prox
    a_old::Tx        # placeholder for previous alpha   
end

function SAGA_prox_state(a, γ::R, av::Tx, z::Tx) where {R,Tx}
    return SAGA_prox_state{R,Tx}(a, γ, av, z, 1, copy(av), copy(av), copy(av), a[1])
end

function Base.iterate(iter::SAGA_prox_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "smoothness parameter absent"
            return nothing
        else
            L_M = maximum(iter.L)
            γ = 1/(5*N*L_M)
        end
    else
        γ = iter.γ # provided γ
    end
    # computing the gradients and updating the table 
    a = Vector{Tx}(undef, 0)
    av = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        av += ∇f/N
        push!(a, iter.x0) # table of x_i
    end
    #initializing the vectors  
    z, ~ = prox(iter.g, (1 - γ) .* iter.x0, γ)
    state = SAGA_prox_state(a, γ, av, z)
    return state, state
end

function Base.iterate(iter::SAGA_prox_iterable{R}, state::SAGA_prox_state{R}) where {R}

    for i = 1:iter.N # for speed in implementation
        state.ind = rand(1:iter.N) # one random number (b=1)
        gradient!(state.∇f_temp, iter.F[state.ind], state.z)
        gradient!(state.temp, iter.F[state.ind], state.a[state.ind])

        @. state.w = state.z - state.γ * (state.∇f_temp - state.temp + state.av)

        state.ind = rand(1:iter.N) # one random number (b=1)
        state.a_old .= state.a[state.ind]
        state.a[state.ind] .= state.z
        
        gradient!(state.∇f_temp, iter.F[state.ind], state.a_old)
        gradient!(state.temp, iter.F[state.ind], state.a[state.ind])
        @. state.av -= (state.∇f_temp - state.temp) / iter.N

        prox!(state.z, iter.g, state.w, state.γ)
    end

    return state, state
end

solution(state::SAGA_prox_state) = state.z
#TODO: minibatch
