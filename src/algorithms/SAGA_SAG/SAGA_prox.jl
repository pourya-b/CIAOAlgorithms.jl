struct SAGA_prox_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    γ::Maybe{R}             # stepsize
    μ::Maybe{R}             # strong convexity moduli
end

mutable struct SAGA_prox_state{R<:Real,Tx,Mx}
    A::Mx                   # table of alphas (needs memory!) cf. alg 2 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization"
    γ::R                    # stepsize 
    av::Tx                  # the running average
    z::Tx                   # the latest iteration
    # some extra placeholders 
    ind::Int                # running idx set 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx                # placeholder for gradients 
    w::Tx                   # input of prox
    a_old::Tx               # placeholder for previous alpha   
end

function SAGA_prox_state(A::Mx, γ::R, av::Tx, z::Tx) where {R,Tx,Mx}
    return SAGA_prox_state{R,Tx,Mx}(A, γ, av, z, 1, copy(av), copy(av), copy(av), A[:,1])
end

function Base.iterate(iter::SAGA_prox_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    n = size(iter.x0)[1]
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "smoothness parameter absent"
            return nothing
        else
            L_M = maximum(iter.L)
            if iter.μ === 0 # non-strongly convex case
                γ = 1/(3*L_M) # according to supp. material of "SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives"
            elseif iter.μ > 0 # strongly convex case (for each of function f_i)
                γ = 1/(2* (iter.μ*N + L_M)) # according to supp. material of "SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives"
            else # noncovnex case
                γ = 1/(5*N*L_M) # according to thm 3 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization", for batch size = 1
            end
        end
    else
        γ = iter.γ # provided γ
    end
    # computing the gradients and updating the table 
    A = zeros(n,N)
    av = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        av += ∇f/N
        A[:,i] = iter.x0
    end

    #initializing the vectors  
    z, ~ = prox(iter.g, iter.x0 - γ * av, γ)
    state = SAGA_prox_state(A, γ, av, z)
    return state, state
end

function Base.iterate(iter::SAGA_prox_iterable{R}, state::SAGA_prox_state{R}) where {R}
    # according to alg 2 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization"
    for i = 1:iter.N # for speed in implementation it is executed for one epoch
        state.ind = rand(1:iter.N) # one random number (b=1)
        gradient!(state.∇f_temp, iter.F[state.ind], state.z)
        gradient!(state.temp, iter.F[state.ind], state.A[:,state.ind])

        state.w .= (state.∇f_temp - state.temp + state.av)
        state.w .*= - state.γ
        state.w .+= state.z  

        state.ind = rand(1:iter.N) # one random number (b=1)
        state.a_old .= state.A[:,state.ind]
        state.A[:,state.ind] .= state.z
        
        gradient!(state.∇f_temp, iter.F[state.ind], state.a_old)
        gradient!(state.temp, iter.F[state.ind], state.z)
        
        state.av .-= (state.∇f_temp - state.temp)/(iter.N)

        prox!(state.z, iter.g, state.w, state.γ)
    end

    return state, state
end

solution(state::SAGA_prox_state) = state.z
