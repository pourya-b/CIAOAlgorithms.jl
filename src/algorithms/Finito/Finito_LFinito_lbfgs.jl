struct FINITO_lbfgs_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg, TH} <: CIAO_iterable
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                    # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8             # to only use one stepsize γ
    batch::Int                # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    H::TH
end

mutable struct FINITO_lbfgs_state{R<:Real,Tx, TH}
    γ::Array{R}             # stepsize parameter
    hat_γ::R                # average γ 
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches 
    H::TH                   # Hessian approx
    # some extra placeholders 
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    zbar::Tx                # bar z 
    zbar_prev::Maybe{Tx}    # bar z previous
    res_zbar::Tx            # v
    res_zbar_prev::Maybe{Tx}# v
    dir::Tx                 # direction
    ∇f_sum::Tx              # for linesearch
    z_trial::Tx             # linesearch candidate
    inds::Array{Int}        # needed for shuffled only! 
    τ::Float64              # number of epochs
end

function FINITO_lbfgs_state(γ::Array{R}, hat_γ::R, av::Tx, ind, d, H::TH) where {R,Tx,TH}
    return FINITO_lbfgs_state{R,Tx,TH}(
        γ,
        hat_γ,
        av,
        ind,
        d,
        H, 
        copy(av),
        copy(av),
        copy(av),
        nothing,
        copy(av),
        nothing,
        copy(av),
        copy(av),
        copy(av),
        collect(1:d),
        1.0
        )
end

function Base.iterate(iter::FINITO_lbfgs_iterable{R}) where {R}
    N = iter.N
    r = iter.batch # batch size 
    # create index sets 
    ind = Vector{Vector{Int}}(undef, 0)
    d = Int(floor(N / r))
    for i = 1:d
        push!(ind, collect(r*(i-1)+1:i*r))
    end
    r * d < N && push!(ind, collect(r*d+1:N))
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            γ = zeros(R, N)
            for i = 1:N
                isa(iter.L, R) ? (γ = fill(iter.α * R(iter.N) / iter.L, (N,))) :
                (γ[i] = iter.α * R(N) / (iter.L[i]))
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) # provided γ
    end
    #initializing the vectors 
    hat_γ = 1 / sum(1 ./ γ)
    av = copy(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        ∇f .*= hat_γ / N
        av .-= ∇f
    end
    state = FINITO_lbfgs_state(γ, hat_γ, av, ind, cld(N, r), iter.H)

    return state, state
end

function Base.iterate(
    iter::FINITO_lbfgs_iterable{R},
    state::FINITO_lbfgs_state{R},
) where {R}
    
    if state.zbar_prev === nothing
        state.zbar_prev = zero(state.z)
        state.res_zbar_prev = zero(state.z)
    end

    # full update 
    # state.zbar, gz = prox(iter.g, state.av, state.hat_γ)
    # envVal = gz
    state.zbar, ~ = prox(iter.g, state.av, state.hat_γ)
    envVal = 0.0
    state.av .= state.zbar
    state.∇f_sum .= zero(state.av)
    for i = 1:iter.N
        state.∇f_temp, fi_z = gradient(iter.F[i], state.zbar) # update the gradient
        state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
        envVal += fi_z / iter.N
        state.∇f_sum .+= state.∇f_temp 
    end
    state.res_zbar, gz = prox(iter.g, state.av, state.hat_γ)
    envVal += gz 
    # state.count += 1

    state.res_zbar .-= state.zbar # \bar v- \bar z 

    if state.zbar_prev !== nothing
        # update metric ##### bug prone  v = -res not res
        update!(state.H, state.zbar - state.zbar_prev, -state.res_zbar +  state.res_zbar_prev) 
        # store vectors for next update
        copyto!(state.zbar_prev, state.zbar)
        copyto!(state.res_zbar_prev, state.res_zbar)
    end

    # println(state.res_zbar)

    mul!(state.dir, state.H, state.res_zbar)


    envVal += real(dot(state.∇f_sum, state.res_zbar)) / iter.N
    envVal += norm(state.res_zbar)^2 / (2 *  state.hat_γ)


    # Compute_direction!(iter,state)   
    # state.dir .=  state.res_zbar # sanity check   

    # println("envelope(bar z) outside is $(envVal)")
    state.τ = 1.0
    # while true  ######## z_line should be removed using some other placeholder
    for i=1:5
        state.z_trial .=  state.zbar .+ (1- state.τ) .* state.res_zbar + state.τ * state.dir

        # compute varphi(z_trial) 
        state.av .= state.z_trial
        state.∇f_sum = zero(state.av)
        envVal_trial = 0
        for i = 1:iter.N
            state.∇f_temp, fi_z = gradient(iter.F[i], state.z_trial) # update the gradient
            state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
            envVal_trial += fi_z / iter.N
            state.∇f_sum .+= state.∇f_temp 
        end
        # state.count += 1
        state.z, gz = prox(iter.g, state.av, state.hat_γ)
        envVal_trial += gz
        state.z .-= state.z_trial # \bar v- \bar z 

        envVal_trial += real(dot(state.∇f_sum, state.z)) / iter.N
        envVal_trial += norm(state.z)^2 / (2 *  state.hat_γ)

        envVal_trial <= envVal + eps(R) && break
        # println("ls backtracked, tau was $(state.τ)")
        state.τ /= 50       ##### bug prone: change in reporting if you change this! ######
    end
    state.zbar .= state.z_trial # of line search

    ###### at the momoent I am computing the next prox twice (it is already computed in the ls) 
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled

    for j in state.inds
        prox!(state.z, iter.g, state.av, state.hat_γ)
        for i in state.ind[j]
            gradient!(state.∇f_temp, iter.F[i], state.zbar) # update the gradient
            state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
            gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
            state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
            state.av .+= (state.hat_γ / state.γ[i]) .* (state.z .- state.zbar)
        end
    end
    # state.count += 1
    return state, state
end

solution(state::FINITO_lbfgs_state) = state.z


epoch_count(state::FINITO_lbfgs_state) = state.τ   # number of epochs is 2+ 1/tau , where 1/tau is from ls 



# # only for logistic loss with g = 0
# function Compute_direction!(iter::FINITO_lbfgs_iterable{R}, state::FINITO_lbfgs_state{R},
# ) where {R}
    
#     n = length(state.z)
#     temp =  similar(state.z, n, n)   # bug prone
#     state.H = zero(temp)
#     for i = 1:iter.N
#         Hessian!(temp, iter.F[i], state.zbar)
#         state.H .+= temp
#     end
#     # state.H ./= iter.N
#     state.dir .=  state.H\(state.res_zbar)
#     state.dir .*= iter.N 
#     state.dir ./= state.hat_γ


#     #     n = size(state.z,1)
#     # state.H .= diagm(0 => ones(n)) .* (iter.F[1].fs[2].lambda * iter.N )    # bug prone
#     # for i = 1:iter.N
#     #     expyx = iter.F[i].fs[1].f.y[1] * real(dot(iter.F[i].fs[1].L, state.zbar))
#     #     # println("exp(Xi) is $(expyx)")
#     #     state.H .+= iter.F[i].fs[1].L' * iter.F[i].fs[1].L .*( expyx/ (1+expyx)^2 ) 
#     # end
#     # # state.H ./= iter.N
#     # state.dir .=  state.H\(state.res_zbar)
#     # state.dir .*= iter.N 
#     # state.dir ./= state.hat_γ
# end




# ##### only for lasso for now! 
# function Compute_direction!(iter::FINITO_lbfgs_iterable{R}, state::FINITO_lbfgs_state{R},
# ) where {R}
    
#     n = size(state.z,1)
#     state.H .= diagm(0 => ones(n))    
#     for i = 1:iter.N # \hat H = 1 - hat_γ/N sum_i nabla f_i(bar z)
#         H_temp, ~ = Hessian(iter.F[i], state.zbar)
#         state.H .-= H_temp .* (state.hat_γ / iter.N)
#     end
#     # state.H ./= iter.N # 

#     mul!(state.∇f_temp, state.H, state.res_zbar) # H(bar z)(bar v - bar z) 

#     # H_temp = similar(H)
#     J_temp, ~ = Jac(iter.g, state.av, state.hat_γ) # jacobian at hat s

#     H_new = diagm(0 => ones(n)) - state.H * J_temp

#     # println(J_temp)
#     println(eigmin(H_new))

#     state.∇f_sum .= H_new\state.∇f_temp    # temp
#     # println(norm(state.∇f_sum, Inf))
#     mul!(state.dir, J_temp, state.∇f_sum)
#     state.dir .+= state.res_zbar
# end




















# only for logistic loss with g = 0 bakcup!
# function Compute_direction!(iter::FINITO_lbfgs_iterable{R}, state::FINITO_lbfgs_state{R},
# ) where {R}
    
#     n = size(state.z,1)
#     state.H .= diagm(0 => ones(n)) .* (iter.F[1].fs[2].lambda * iter.N )    # bug prone
#     for i = 1:iter.N
#         expyx = iter.F[i].fs[1].f.y[1] * real(dot(iter.F[i].fs[1].L, state.zbar))
#         # println("exp(Xi) is $(expyx)")
#         state.H .+= iter.F[i].fs[1].L' * iter.F[i].fs[1].L .*( expyx/ (1+expyx)^2 ) 
#     end
#     # state.H ./= iter.N
#     state.dir .=  state.H\(state.res_zbar)
#     state.dir .*= iter.N 
#     state.dir ./= state.hat_γ
# end
