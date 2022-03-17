struct SGD_prox_DNN_iterable{R<:Real,Tg,Tf,Tp}
    F::Union{Array{Tf},Tf}  # smooth term  
    g::Tg                   # nonsmooth term 
    dnn_params::Tp          # initial point
    N::Int64                # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    γ::Maybe{R}             # stepsize 
    diminishing::Bool       # diminishing version (diminishing stepsize)
    η0::R                   # for the stepsize: η0/(η_tilde + epoch_counter), if diminishing
    η_tilde::R              # for the stepsize: η0/(η_tilde + epoch_counter), if diminishing
    data::Tuple
    DNN_config!
end

mutable struct SGD_prox_DNN_state{R<:Real,Tx,Mx}
    γ::Maybe{R}             # stepsize 
    z::Tx
    cind::Int               # current interation index
    idxr::Int               # current index
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
    temp_x::Mx
    temp_y::Tx
end

function SGD_prox_DNN_state(γ::Maybe{R}, z::Tx, cind, temp_x::Mx, temp_y) where {R,Tx,Mx}
    return SGD_prox_DNN_state{R,Tx,Mx}(γ, z, cind, Int(0), copy(z), copy(z), temp_x, temp_y)
end

function Base.iterate(iter::SGD_prox_DNN_iterable{R}) where {R}
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
        γ = 0.0
    end

    # initializing
    cind = 0
    z = iter.DNN_config!()
    temp_x = zeros(size(iter.dnn_params[1],2))
    temp_y = zeros(size(iter.dnn_params[length(iter.dnn_params)],1))
    state = SGD_prox_DNN_state(γ, z, cind, temp_x, temp_y)
    return state, state
end

function Base.iterate(iter::SGD_prox_DNN_iterable{R}, state::SGD_prox_DNN_state{R}) where {R}
    # The inner cycle
    for i=1:iter.N # for speed in implementation it is executed for one epoch
        state.cind += 1
        if iter.diminishing # according to "Stochastic first-and zeroth-order methods for nonconvex stochastic programming".
            state.γ = iter.η0/(1 + iter.η_tilde * floor(state.cind/iter.N))
        end

        state.temp_x .= iter.data[1][:,i]
        state.temp_y .= iter.data[2][:,i]
        
        gs = Flux.gradient(iter.dnn_params) do # gradient of F[i] wrt dnn_params, gs is an object of Zygote
            iter.F(state.temp_x,state.temp_y)
        end
        state.∇f_temp .= iter.DNN_config!(gs=gs) # vectorized gs (state.∇f_temp = gs)

        state.∇f_temp .*= - state.γ
        state.∇f_temp ./= iter.N 
        state.∇f_temp .+= state.z

        CIAOAlgorithms.prox!(state.z, iter.g, state.∇f_temp, state.γ)
        iter.DNN_config!(gs=state.z) # setting the dnn_params to state.z
    end

    return state, state
end

solution(state::SGD_prox_DNN_state) = state.z
