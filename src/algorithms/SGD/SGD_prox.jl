struct SGD_prox_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    μ::Maybe{Union{Array{R},R}}  # convexity moduli of the gradients
    γ::Maybe{R}             # stepsize 
    diminishing::Bool       # diminishing stepsize
    η0::R                   # for the stepsize: η0/(η_tilde + epoch_counter), if diminishing
    η_tilde::R              # for the stepsize: η0/(η_tilde + epoch_counter), if diminishing
end

mutable struct SGD_prox_state{R<:Real,Tx}
    γ::Maybe{R}             # stepsize 
    z::Tx
    cind::Int               # current interation index
    idxr::Int               # current index
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
end

function SGD_prox_state(γ::Maybe{R}, z::Tx, cind) where {R,Tx}
    return SGD_prox_state{R,Tx}(γ, z, cind, Int(0), copy(z), copy(z))
end

function Base.iterate(iter::SGD_prox_iterable{R}) where {R}
    N = iter.N
    ind = collect(1:N)
    γ = iter.γ
    # updating the stepsize
    if ~iter.diminishing 
        if iter.γ === nothing
            if iter.L === nothing
                @warn "smoothness or convexity parameter absent"
                return nothing
            else
                L_M = maximum(iter.L)
                γ = 1/(2*L_M)
            end
        else
            γ = iter.γ # provided γ
        end
    end

    if iter.diminishing
        println("diminishing stepsize version")
    end

    # initializing the vectors 
    ∇f_temp = zero(iter.x0)
    z = zero(iter.x0)
    z .= iter.x0
    cind = 0
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], z)
        if iter.diminishing
            cind += 1
            γ = iter.η0/(iter.η_tilde + cind/N)
        end
        ∇f ./= N
        ∇f .*= - γ    
        ∇f .+= z
        CIAOAlgorithms.prox!(z, iter.g, ∇f, γ)
    end

    state = SGD_prox_state(γ, z, cind)
    return state, state
end

function Base.iterate(iter::SGD_prox_iterable{R}, state::SGD_prox_state{R}) where {R} 
    # The inner cycle
    for i=1:iter.N # for speed in implementation it is executed for one epoch
        state.cind += 1
        if iter.diminishing
            state.γ = iter.η0/(iter.η_tilde + state.cind/iter.N)
        end

        gradient!(state.∇f_temp, iter.F[i], state.z)

        state.∇f_temp .*= - state.γ
        state.∇f_temp ./= iter.N 
        state.∇f_temp .+= state.z

        CIAOAlgorithms.prox!(state.z, iter.g, state.∇f_temp, state.γ)
    end

    return state, state
end


solution(state::SGD_prox_state) = state.z
