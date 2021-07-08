struct FINITO_Newton_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                    # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8             # to only use one stepsize γ
    batch::Int                # batch size
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct FINITO_Newton_state{R<:Real,Tx}
    γ::Array{R}             # stepsize parameter
    hat_γ::R                # average γ 
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches 
    # some extra placeholders 
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    z_full::Tx
    v::Tx                   # v
    dir::Tx                 # direction
    ∇f_sum::Tx              # for linesearch
    z_trial::Tx             # linesearch candidate
    inds::Array{Int}        # needed for shuffled only! 
    τ::Float64              # number of epochs
    H::Array{Float64,2}     # Hessian
end

function FINITO_Newton_state(γ::Array{R}, hat_γ::R, av::Tx, ind, d) where {R,Tx}
    return FINITO_Newton_state{R,Tx}(
        γ,
        hat_γ,
        av,
        ind,
        d,
        copy(av),
        copy(av),
        copy(av),
        copy(av),
        copy(av),
        copy(av),
        copy(av),
        collect(1:d),
        1.0, 
        zeros(size(av,1),size(av,1))
    )
end

function Base.iterate(iter::FINITO_Newton_iterable{R}) where {R}
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
    state = FINITO_Newton_state(γ, hat_γ, av, ind, cld(N, r))

    return state, state
end

function Base.iterate(
    iter::FINITO_Newton_iterable{R},
    state::FINITO_Newton_state{R},
) where {R}
    # full update 
    state.z_full, gz = prox(iter.g, state.av, state.hat_γ)
    envVal = gz
    state.av .= state.z_full
    state.∇f_sum .= zero(state.av)
    for i = 1:iter.N
        state.∇f_temp, fi_z = gradient(iter.F[i], state.z_full) # update the gradient
        state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
        envVal += fi_z / iter.N
        state.∇f_sum .+= state.∇f_temp 
    end
    prox!(state.v, iter.g, state.av, state.hat_γ)

    # state.count += 1

    state.v .-= state.z_full # \bar v- \bar z 

    envVal += real(dot(state.∇f_sum ./ iter.N, state.v))
    envVal += norm(state.v)^2 / (2 *  state.hat_γ)


    Compute_direction!(iter,state)   
    # state.dir .=  state.v # sanity check   

    # println("envelope(bar z) outside is $(envVal)")
    state.τ = 1.0
    while true  ######## z_line should be removed using some other placeholder
        state.z_trial .=  state.z_full .+ (1- state.τ) .* state.v + state.τ * state.dir

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

        envVal_trial += real(dot(state.∇f_sum ./ iter.N, state.z))
        envVal_trial += norm(state.z)^2 / (2 *  state.hat_γ)

        envVal_trial <= envVal + eps(R) && break
        # println("ls backtracked, tau was $(state.τ)")
        state.τ /= 100       ##### bug prone: change in reporting if you change this! ######
    end
    state.z_full .= state.z_trial # of line search

    ###### at the momoent I am computing the next prox twice (it is already computed in the ls) 
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled

    for j in state.inds
        prox!(state.z, iter.g, state.av, state.hat_γ)
        for i in state.ind[j]
            gradient!(state.∇f_temp, iter.F[i], state.z_full) # update the gradient
            state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
            gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
            state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
            state.av .+= (state.hat_γ / state.γ[i]) .* (state.z .- state.z_full)
        end
    end
    # state.count += 1
    return state, state
end

solution(state::FINITO_Newton_state) = state.z


epoch_count(state::FINITO_Newton_state) = state.τ   # number of epochs is 2+ 1/tau , where 1/tau is from ls 



# only for logistic loss with g = 0
function Compute_direction!(iter::FINITO_Newton_iterable{R}, state::FINITO_Newton_state{R},
) where {R}
    
    n = length(state.z)
    temp =  similar(state.z, n, n)   # bug prone
    state.H = zero(temp)
    for i = 1:iter.N
        Hessian!(temp, iter.F[i], state.z_full)
        state.H .+= temp
    end
    # state.H ./= iter.N
    state.dir .=  state.H\(state.v)
    state.dir .*= iter.N 
    state.dir ./= state.hat_γ


    #     n = size(state.z,1)
    # state.H .= diagm(0 => ones(n)) .* (iter.F[1].fs[2].lambda * iter.N )    # bug prone
    # for i = 1:iter.N
    #     expyx = iter.F[i].fs[1].f.y[1] * real(dot(iter.F[i].fs[1].L, state.z_full))
    #     # println("exp(Xi) is $(expyx)")
    #     state.H .+= iter.F[i].fs[1].L' * iter.F[i].fs[1].L .*( expyx/ (1+expyx)^2 ) 
    # end
    # # state.H ./= iter.N
    # state.dir .=  state.H\(state.v)
    # state.dir .*= iter.N 
    # state.dir ./= state.hat_γ
end




# ##### only for lasso for now! 
# function Compute_direction!(iter::FINITO_Newton_iterable{R}, state::FINITO_Newton_state{R},
# ) where {R}
    
#     n = size(state.z,1)
#     state.H .= diagm(0 => ones(n))    
#     for i = 1:iter.N # \hat H = 1 - hat_γ/N sum_i nabla f_i(bar z)
#         H_temp, ~ = Hessian(iter.F[i], state.z_full)
#         state.H .-= H_temp .* (state.hat_γ / iter.N)
#     end
#     # state.H ./= iter.N # 

#     mul!(state.∇f_temp, state.H, state.v) # H(bar z)(bar v - bar z) 

#     # H_temp = similar(H)
#     J_temp, ~ = Jac(iter.g, state.av, state.hat_γ) # jacobian at hat s

#     H_new = diagm(0 => ones(n)) - state.H * J_temp

#     # println(J_temp)
#     println(eigmin(H_new))

#     state.∇f_sum .= H_new\state.∇f_temp    # temp
#     # println(norm(state.∇f_sum, Inf))
#     mul!(state.dir, J_temp, state.∇f_sum)
#     state.dir .+= state.v
# end




















# only for logistic loss with g = 0 bakcup!
# function Compute_direction!(iter::FINITO_Newton_iterable{R}, state::FINITO_Newton_state{R},
# ) where {R}
    
#     n = size(state.z,1)
#     state.H .= diagm(0 => ones(n)) .* (iter.F[1].fs[2].lambda * iter.N )    # bug prone
#     for i = 1:iter.N
#         expyx = iter.F[i].fs[1].f.y[1] * real(dot(iter.F[i].fs[1].L, state.z_full))
#         # println("exp(Xi) is $(expyx)")
#         state.H .+= iter.F[i].fs[1].L' * iter.F[i].fs[1].L .*( expyx/ (1+expyx)^2 ) 
#     end
#     # state.H ./= iter.N
#     state.dir .=  state.H\(state.v)
#     state.dir .*= iter.N 
#     state.dir ./= state.hat_γ
# end
