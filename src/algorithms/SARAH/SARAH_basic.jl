struct SARAH_basic_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf}
    F::Array{Tf}            # smooth term - array of f_i s 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of ∇f_i	
    μ::Maybe{Union{Array{R},R}}  # convexity moduli of the gradients
    γ::Maybe{R}             # stepsize (a single scalar)
    m::Maybe{Int}           # number of inner loop updates
end

mutable struct SARAH_basic_state{R<:Real,Tx}
    γ::R                    # stepsize 
    m::Maybe{Int}           # number of inner loop updates
    av::Tx                  # average of ∇f_i s in outer loop
    w::Tx                   # w_{t-1} in the inner loop
    w_plus::Tx              # w_{t} in the inner loop
    ind::Array{Int}         # running idx set (1 to N)
    
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx                # placeholder for gradients 
end

function SARAH_basic_state(γ::R, m, av::Tx, w::Tx, w_plus::Tx, ind) where {R,Tx}
    return SARAH_basic_state{R,Tx}(γ, m, av, w, w_plus, ind, copy(av), copy(av))
end

function Base.iterate(iter::SARAH_basic_iterable{R}) where {R} #? we doesn't need other types like C, Tx, ...?
    N = iter.N
    m = iter.m
    ind = collect(1:N) # a vector of 1 to N 
    L_M = maximum(iter.L)

    # updating the stepsize 
    if iter.γ === nothing
        if iter.μ === nothing # means it is nonconvex
            if iter.L === nothing 
                @warn "smoothness parameter absent"
                return nothing
            else
                γ = 0.5 / L_M 
            end
        elseif maximum(iter.μ) == 0 # convex
            @warn "convex problem"
            γ = 0.5 / L_M 
        else
            @warn "strongly convex problem"
            γ = 0.5 / L_M 
        end
    else
        γ = iter.γ # provided γ
    end
    # initializing the vectors 
    
    av = zero(iter.x0) # mu_tilde in the outer loop
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0) 
        ∇f ./= N
        av .+= ∇f 
    end
    w = iter.x0
    w_plus = w - γ * av

    state = SARAH_basic_state(γ, m, av, w, w_plus, ind)
    return state, state
end

function Base.iterate(iter::SARAH_basic_iterable{R}, state::SARAH_basic_state{R}) where {R}
    # The inner cycle
    for i in rand(state.ind, state.m) # Uniformly randomly pick one index \in [N] (with replacement) in each iteration
        gradient!(state.temp, iter.F[i], state.w_plus) 
        gradient!(state.∇f_temp, iter.F[i], state.w)
        state.av .+= state.temp
        state.av .-= state.∇f_temp

        state.temp .= state.av
        state.temp .*= state.γ
        state.w .= state.w_plus
        state.w_plus .-= state.temp
    end
    
    # full update 	
    state.w .= state.w_plus
    state.av .= zero(state.w)  # for next iterate 
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.F[i], state.w)
        state.∇f_temp ./= iter.N
        state.av .+= state.∇f_temp
    end
    state.temp .= state.av
    state.temp .*= state.γ
    state.w_plus .-= state.temp

    return state, state
end

solution(state::SARAH_basic_state) = state.w_plus

## TO DO:
## provide appropriate stepsizes for different convexity levels
## provide the corresponding references for each stepsize 