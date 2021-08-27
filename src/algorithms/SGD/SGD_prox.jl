struct SGD_prox_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    μ::Maybe{Union{Array{R},R}}  # convexity moduli of the gradients
    γ::Maybe{R}             # stepsize 
    plus::Bool              # plus version
end

mutable struct SGD_prox_state{R<:Real,Tx}
    γ::R                    # stepsize 
    z::Tx
    ind::Array{Int}         # running idx set 
    cind::Int               # current interation index
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
end

function SGD_prox_state(γ::R, z::Tx, ind) where {R,Tx}
    return SGD_prox_state{R,Tx}(γ, z, ind, Int(1), copy(z), copy(z))
end

function Base.iterate(iter::SGD_prox_iterable{R}) where {R}
    N = iter.N
    ind = collect(1:N)
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "smoothness or convexity parameter absent"
            return nothing
        else
            L_M = maximum(iter.L)
            γ = 1/ (2*L_M)
        end
    else
        γ = iter.γ # provided γ
    end
    if iter.plus
        γ = 0.1/(1+1/N)
        println("plus version")
    end
    # initializing the vectors 
    ∇f_temp = zero(iter.x0)
    temp = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        ∇f ./= N
        ∇f_temp .+= ∇f
    end
    ∇f_temp .*= - γ
    ∇f_temp .+= iter.x0
    CIAOAlgorithms.prox!(temp, iter.g, ∇f_temp, γ)
    state = SGD_prox_state(γ, temp, ind)
    return state, state
end

function Base.iterate(iter::SGD_prox_iterable{R}, state::SGD_prox_state{R}) where {R}
    # The inner cycle
    state.cind += 1
    if iter.plus
        state.γ = 0.1/(1+state.cind/iter.N)
    end
    state.temp .= zero(state.z)
    for i in state.ind
        gradient!(state.∇f_temp, iter.F[i], state.z)
        state.temp .+= state.∇f_temp
    end

    state.temp .*= - state.γ
    state.temp ./= iter.N 
    state.temp .+= state.z

    CIAOAlgorithms.prox!(state.z, iter.g, state.temp, state.γ)

    return state, state
end


solution(state::SGD_prox_state) = state.z
