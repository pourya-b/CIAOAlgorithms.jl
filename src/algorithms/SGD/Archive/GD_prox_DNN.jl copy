struct GD_prox_DNN_iterable{R<:Real,Tg,Tf,Tp}
    F::Union{Array{Tf},Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    opt_params::Tp                  # initial point
    N::Int64                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    μ::Maybe{Union{Array{R},R}}  # convexity moduli of the gradients
    γ::Maybe{R}             # stepsize 
    plus::Bool              # plus version (diminishing stepsize)
    η0::R
    η_tilde::R
    data::Tuple
    DNN_config!
end

mutable struct GD_prox_DNN_state{R<:Real,Tx,Mx}
    γ::R                    # stepsize 
    z::Tx
    cind::Int               # current interation index
    idxr::Int               # current index
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
    temp_x::Mx
    temp_y::Tx
end

function GD_prox_DNN_state(γ::R, z::Tx, cind, temp_x::Mx, temp_y) where {R,Tx,Mx}
    return GD_prox_DNN_state{R,Tx,Mx}(γ, z, cind, Int(0), copy(z), copy(z), temp_x, temp_y)
end

function Base.iterate(iter::GD_prox_DNN_iterable{R}) where {R}
    println("started")
    N = iter.N
    ind = collect(1:N)
    # updating the stepsize 
    if iter.γ === nothing && !iter.plus
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
        println("diminishing stepsize version")
        γ = 0.0
    end
    # initializing
    cind = 0
    z = iter.DNN_config!()
    temp_x = zeros(size(iter.opt_params[1],2))
    temp_y = zeros(size(iter.opt_params[length(iter.opt_params)],1))
    state = GD_prox_DNN_state(γ, z, cind, temp_x, temp_y)
    return state, state
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

function Base.iterate(iter::GD_prox_DNN_iterable{R}, state::GD_prox_DNN_state{R}) where {R}
    # The inner cycle
    state.cind += 1
    if iter.plus
        state.γ = iter.η0/(1 + iter.η_tilde * floor(state.cind/iter.N))
    end
    # if mod(state.cind,100) == 0
    #     println("iter: $(i)")
    #     flush(stdout)
    # end

    # state.tupl .= (iter.data[1][:,i],iter.data[2][:,i])
    # state.temp_x .= iter.data[1][:,i]
    # state.temp_y .= iter.data[2][:,i]
    # gradient!(state.∇f_temp, iter.F[i], state.z)
    # state.∇f_temp .= (Flux.gradient(() -> iter.F[i](state.z), params(state.z)))[state.z][:,1]
    
    gs = Flux.gradient(iter.opt_params) do # sampled gradient
    iter.F(batchmemaybe(iter.data)...)
    # iter.F(batchmemaybe(tupl)...)
    end
    state.∇f_temp .= iter.DNN_config!(gs=gs)
    state.z .= iter.DNN_config!()

    state.∇f_temp .*= - state.γ
    # state.∇f_temp ./= iter.N 
    state.∇f_temp .+= state.z

    CIAOAlgorithms.prox!(state.z, iter.g, state.∇f_temp, state.γ)
    iter.DNN_config!(gs=state.z)

    return state, state
end

solution(state::GD_prox_DNN_state) = state.z
