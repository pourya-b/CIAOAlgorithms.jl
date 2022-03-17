# SPIRAL for DNN training
struct FINITO_lbfgs_iterable_DNN{R<:Real,Tp,Tf,Tg,TH} <: CIAO_iterable
    F::Tf                   # smooth term  
    g::Tg                   # nonsmooth term 
    dnn_params::Tp                  # initial point
    N::Int                   # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    β::R                    # ls division parameter for τ
    α::R                    # in (0, 1), e.g.: 0.99
    H::TH                   # LBFGS struct
    data::Tuple             # data
    DNN_config!             # a function
end

mutable struct FINITO_lbfgs_state_DNN{R<:Real,Tx, TH}
    γ::Array{R}             # stepsize parameter
    hat_γ::R                # average γ 
    av::Tx                  # the running average (vector s)
    H::TH                   # Hessian approx (LBFGS struct)
    # some extra placeholders 
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    zbar::Tx                # bar z = z^k
    zbar_prev::Maybe{Tx}    # bar z previous
    res_zbar::Tx            # v^k = prox(z^k)
    res_zbar_prev::Maybe{Tx}# v previous
    dir::Tx                 # quasi_Newton direction
    ∇f_sum::Tx              # placeholder for gradient sum
    z_trial::Tx             # linesearch candidate
    τ::Float64              # interpolation parameter between the quasi-Newton direction and the nominal step 
end

function FINITO_lbfgs_state_DNN(γ::Array{R}, hat_γ::R, av::Tx, H::TH) where {R,Tx,TH}
    return FINITO_lbfgs_state_DNN{R,Tx,TH}(
        γ,
        hat_γ,
        av,
        H, 
        copy(av),
        copy(av),
        copy(av),
        nothing, # zbar_prev
        copy(av),
        nothing,
        copy(av),
        copy(av),
        copy(av),
        1.0
        )
end

function Base.iterate(iter::FINITO_lbfgs_iterable_DNN{R}) where {R}
    N = iter.N
 
    if iter.γ === nothing
        @warn "--> single stepsize parameter absent"
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) # provided γ
    end

    #initializing the vectors 
    hat_γ = 1 / sum(1 ./ γ)
    av = copy(iter.DNN_config!())

    gs = Flux.gradient(iter.dnn_params) do # dnn_params is the DNN parameters (Zygote object)
        iter.F(batchmemaybe(iter.data)...)
    end
    ∇f = iter.DNN_config!(gs=gs) # F is already divided by N, as it is mean function
    ∇f .*= hat_γ
    av .-= ∇f

    state = FINITO_lbfgs_state_DNN(γ, hat_γ, av, iter.H)

    return state, state
end

function Base.iterate(
    iter::FINITO_lbfgs_iterable_DNN{R},
    state::FINITO_lbfgs_state_DNN{R},
) where {R}
    
    if state.zbar_prev === nothing # for lbfgs updates
        state.zbar_prev = zero(state.z)
        state.res_zbar_prev = zero(state.z)
    end

    # full update 
    state.zbar, ~ = prox(iter.g, state.av, state.hat_γ) # z^k in Alg. 1&2

    # envVal = 0.0 # envelope value (lyapunov function) L(v^k,z^k)
    # state.av .= state.zbar # vector \bar s
    # state.∇f_sum .= zero(state.av)
    # for i = 1:iter.N
    #     state.∇f_temp, fi_z = gradient(iter.F[i], state.zbar) # update the gradient
    #     state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
    #     envVal += fi_z / iter.N # build by (1.1) - first part
    #     state.∇f_sum .+= state.∇f_temp 
    # end
    
    iter.DNN_config!(gs=state.zbar)
    gs = Flux.gradient(iter.dnn_params) do # sampled gradient
        iter.F(batchmemaybe(iter.data)...)
    end
    state.∇f_sum .= iter.DNN_config!(gs=gs) * iter.N
    state.av .= state.zbar .- (state.hat_γ / iter.N) .* state.∇f_sum
    envVal = iter.F(batchmemaybe(iter.data)...) 

    
    state.res_zbar, gz = prox(iter.g, state.av, state.hat_γ) # v^k
    envVal += gz # gz = g(v^k)

    state.res_zbar .-= state.zbar # v- z 

    # update lbfgs
    update!(state.H, state.zbar - state.zbar_prev, -state.res_zbar +  state.res_zbar_prev) # to update H by lbfgs! (H, s-s_pre, y-y_pre)
    # store vectors for next update
    copyto!(state.zbar_prev, state.zbar)
    copyto!(state.res_zbar_prev, state.res_zbar)
    mul!(state.dir, state.H, state.res_zbar) # updating the quasi-Newton direction

    envVal += real(dot(state.∇f_sum, state.res_zbar)) / iter.N # envelope value (lyapunov function) L(v^k,z^k)
    envVal += norm(state.res_zbar)^2 / (2 *  state.hat_γ)

    state.τ = 1.0 
    for i=1:5 # backtracking on τ
        state.z_trial .=  state.zbar .+ (1- state.τ) .* state.res_zbar + state.τ * state.dir # u^k
        
        # state.av .= state.z_trial # vector \tilde s
        # state.∇f_sum = zero(state.av)
        # envVal_trial = 0 # envelope value (lyapunov function) L(y^k,u^k)
        # for i = 1:iter.N
        #     state.∇f_temp, fi_z = gradient(iter.F[i], state.z_trial) # update the gradient
        #     state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
        #     envVal_trial += fi_z / iter.N
        #     state.∇f_sum .+= state.∇f_temp 
        # end

        iter.DNN_config!(gs=state.z_trial)
        gs = Flux.gradient(iter.dnn_params) do # sampled gradient
            iter.F(batchmemaybe(iter.data)...)
        end
        state.∇f_sum .= iter.DNN_config!(gs=gs) * iter.N
        state.av .= state.z_trial .- (state.hat_γ / iter.N) .* state.∇f_sum
        envVal_trial = iter.F(batchmemaybe(iter.data)...) 

        state.z, gz = prox(iter.g, state.av, state.hat_γ) # y^k
        envVal_trial += gz
        state.z .-= state.z_trial # y^k - u^k

        envVal_trial += real(dot(state.∇f_sum, state.z)) / iter.N # envelope value (lyapunov function) L(y^k,u^k)
        envVal_trial += norm(state.z)^2 / (2 *  state.hat_γ)

        envVal_trial <= envVal + eps(R) && break # descent on the envelope function (Table 1, 5e) # here it seems accurate precisions result in better results. No  stabiliry issues are seen.
        state.τ *= iter.β   # backtracking on τ
        println("ls on τ")
    end
    state.zbar .= state.z_trial # u^k

    for i in iter.N # batch indices
        temp_x = iter.data[1][:,i]
        temp_y = iter.data[2][:,i]

        prox!(state.z, iter.g, state.av, state.hat_γ) # \tilde z^k_i, state.av (s^k) is already calculated for envVal

        # gradient!(state.∇f_temp, iter.F[i], state.zbar) # the gradient for u^k
        iter.DNN_config!(gs=state.zbar)
        gs = Flux.gradient(iter.dnn_params) do # sampled gradient
            iter.F(temp_x,temp_y)
        end
        state.∇f_temp .= iter.DNN_config!(gs=gs)
        state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp 

        # gradient!(state.∇f_temp, iter.F[i], state.z) # the gradient for \tilde z^k_i
        iter.DNN_config!(gs=state.z)
        gs = Flux.gradient(iter.dnn_params) do # sampled gradient
            iter.F(temp_x,temp_y)
        end
        state.∇f_temp .= iter.DNN_config!(gs=gs)
        state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp

        state.av .+= (state.hat_γ / state.γ[i]) .* (state.z .- state.zbar) # updating the vector s^k
    end

    return state, state
end

solution(state::FINITO_lbfgs_state_DNN) = state.z
epoch_count(state::FINITO_lbfgs_state_DNN) = state.τ  # number of epochs is epoch_per_iter + log_β(τ) , where tau is from ls and epoch_per_iter is 3 or 4. Refer to it_counter function in utilities.jl
