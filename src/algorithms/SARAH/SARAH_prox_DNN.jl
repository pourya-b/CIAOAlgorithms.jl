struct SARAH_prox_DNN_iterable{R<:Real,Tf,Tg,Tp}
    F::Union{Array{Tf},Tf}            # smooth term - array of f_i s 
    g::Tg                   # nonsmooth part
    dnn_params::Tp                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of ∇f_i	
    γ::Maybe{R}             # stepsize (a single scalar)
    m::Maybe{Int}           # number of inner loop updates
    ꞵ::Maybe{R}             # a scalar momentum 
    data::Tuple
    DNN_config!
end

mutable struct SARAH_prox_DNN_state{R<:Real,Tx}
    γ::R                    # stepsize 
    m::Maybe{Int}           # number of inner loop updates
    ꞵ::R                    # momentum 
    av::Tx                  # average of ∇f_i s in outer loop
    w::Tx                   # w_{t-1} in the inner loop
    w_plus::Tx              # w_{t} in the inner loop
    ind::Array{Int}         # running idx set (1 to N)
    
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx                # placeholder for gradients 
end

function SARAH_prox_DNN_state(γ::R, m, ꞵ::R, av::Tx, w::Tx, w_plus::Tx, ind) where {R,Tx}
    return SARAH_prox_DNN_state{R,Tx}(γ, m, ꞵ, av, w, w_plus, ind, copy(av), copy(av))
end

function Base.iterate(iter::SARAH_prox_DNN_iterable{R}) where {R} 
    N = iter.N
    m = iter.m
    batch = 1
    ω = (3*(N-batch))/(2*batch*(N-1))
    ind = collect(1:N) # a vector of 1 to N 

    # updating the stepsize 
    if iter.γ === nothing
        γ = (2*sqrt(ω * m)) / (4*sqrt(ω * m) + 1) # based on thm 8 of "ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization"
    else
        γ = iter.γ # provided γ
    end

    if iter.ꞵ === nothing
        if iter.L === nothing 
            @warn "smoothness parameter absent"
            return nothing
        else
            L_M = maximum(iter.L)
            ꞵ = 1/(L_M * sqrt(ω * m)) # based on thm 8 of "ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization"
        end
    else
        ꞵ = iter.ꞵ
    end

    # initializing the vectors 
    x0 = iter.DNN_config!()
    # av = zero(x0) 
    w_plus = zero(x0)
    # for i = 1:N
    #     ∇f = (Flux.gradient(() -> iter.F(i,iter.x0), params(iter.x0)))[iter.x0][:,1]
    #     # ∇f, ~ = gradient(iter.F[i], iter.x0) 
    #     ∇f ./= N
    #     av .+= ∇f 
    # end
    gs = Flux.gradient(iter.dnn_params) do # sampled gradient
        iter.F(batchmemaybe(iter.data)...)
    end
    av = iter.DNN_config!(gs=gs)
    w = copy(x0)
    CIAOAlgorithms.prox!(w_plus, iter.g, w - γ * av, γ)
    w_plus .*= ꞵ
    w_plus .+= (1 - ꞵ) * w

    state = SARAH_prox_DNN_state(γ, m, ꞵ, av, w, w_plus, ind)
    return state, state
end

function Base.iterate(iter::SARAH_prox_DNN_iterable{R}, state::SARAH_prox_DNN_state{R}) where {R}
    # The inner cycle
    for i in rand(state.ind, state.m)
        temp_x = iter.data[1][:,i]
        temp_y = iter.data[2][:,i]
        # gradient!(state.temp, iter.F[i], state.w_plus) 
        # state.temp .= (Flux.gradient(() -> iter.F(i,state.w_plus), params(state.w_plus)))[state.w_plus][:,1]
        iter.DNN_config!(gs=state.w_plus)
        gs = Flux.gradient(iter.dnn_params) do # sampled gradient
            iter.F(temp_x,temp_y)
        end
        state.temp .= iter.DNN_config!(gs=gs)

        # gradient!(state.∇f_temp, iter.F[i], state.w)
        # state.∇f_temp .= (Flux.gradient(() -> iter.F(i,state.w), params(state.w)))[state.w][:,1]
        iter.DNN_config!(gs=state.w)
        gs = Flux.gradient(iter.dnn_params) do # sampled gradient
            iter.F(temp_x,temp_y)
        end
        state.∇f_temp .= iter.DNN_config!(gs=gs)

        state.av .+= state.temp
        state.av .-= state.∇f_temp

        state.temp .= state.av
        state.temp .*= state.γ
        state.temp .*= -1
        state.temp .+= state.w_plus
        state.w .= state.w_plus
        CIAOAlgorithms.prox!(state.w_plus, iter.g, state.temp, state.γ)
        state.w_plus .*= state.ꞵ
        state.temp .= state.w
        state.temp .*= 1-state.ꞵ
        state.w_plus .+= state.temp
    end
    # full update 	
    state.w .= state.w_plus
    # state.av .= zero(state.w)  # for next iterate 
    # for i = 1:iter.N
    #     # gradient!(state.∇f_temp, iter.F[i], state.w)
    #     state.∇f_temp .= (Flux.gradient(() -> iter.F(i,state.w), params(state.w)))[state.w][:,1]
    #     state.∇f_temp ./= iter.N
    #     state.av .+= state.∇f_temp
    # end
    iter.DNN_config!(gs=state.w)
    gs = Flux.gradient(iter.dnn_params) do # sampled gradient
        iter.F(batchmemaybe(iter.data)...)
    end
    state.av .= iter.DNN_config!(gs=gs)
    
    state.temp .= state.av
    state.temp .*= state.γ
    state.temp .*= -1
    state.temp .+= state.w
    CIAOAlgorithms.prox!(state.w_plus, iter.g, state.temp, state.γ)
    state.w_plus .*= state.ꞵ
    state.temp .= state.w
    state.temp .*= 1-state.ꞵ
    state.w_plus .+= state.temp

    return state, state
end

solution(state::SARAH_prox_DNN_state) = state.w_plus
