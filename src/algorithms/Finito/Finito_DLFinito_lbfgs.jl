struct FINITO_DFlbfgs_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg, TH} <: CIAO_iterable
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                    # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8             # to only use one stepsize γ
    batch::Int                # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    H::TH                   # for lbfgs (lbfgs type)
    inseq::Int              # extra 'memory length' needed = length of inner seq (excluding z)   # exm: 1231231234234234... as memory = 3
    rep::Int                # number of reps for each index   # how many times 123 is repeated
end

mutable struct FINITO_DFlbfgs_state{R<:Real,Tx, TH}
    γ::Array{R}             # stepsize parameter
    hat_γ::R                # average γ 
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches 
    H::TH                   # Hessian approx
    z_M::Vector{Tx}         # palceholder for z vectors 
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

function FINITO_DFlbfgs_state(γ::Array{R}, hat_γ::R, av::Tx, ind, d, H::TH, z_M) where {R,Tx,TH}
    return FINITO_DFlbfgs_state{R,Tx,TH}(
        γ,
        hat_γ,
        av,
        ind,
        d,
        H, 
        z_M,
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

function Base.iterate(iter::FINITO_DFlbfgs_iterable{R}) where {R}
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
    z_M = [copy(av) for i = 1:(iter.inseq+1)]


    state = FINITO_DFlbfgs_state(γ, hat_γ, av, ind, cld(N, r), iter.H, z_M)

    return state, state
end

function Base.iterate(
    iter::FINITO_DFlbfgs_iterable{R},
    state::FINITO_DFlbfgs_state{R},
) where {R}
    
    if state.zbar_prev === nothing
        state.zbar_prev = zero(state.z)
        state.res_zbar_prev = zero(state.z)
    end

    # full update 
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

    if state.zbar_prev !== nothing # for lbfgs
        # update metric ##### bug prone  v = -res not res
        update!(state.H, state.zbar - state.zbar_prev, -state.res_zbar +  state.res_zbar_prev) 
        # store vectors for next update
        copyto!(state.zbar_prev, state.zbar)
        copyto!(state.res_zbar_prev, state.res_zbar)
    end

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

    # println("rep is $(iter.rep)")
    Ind = Index_iterable(1, iter.inseq, iter.rep, iter.N) # in CIAO utilities

    for (it, ind) in Iterators.enumerate(Ind) # here we don't have batches
        # println(it)
        if mod(it, iter.inseq * iter.rep) == 1
            # println("------new cycle starts-----")
            for ell in 1:(iter.inseq+1)       # for initializing the inner loop
                copyto!(state.z_M[ell], state.zbar) # resetting with zbar: x^full
            end             
        end
        println(ind) 
        prox!(state.z_M[ind.m_n], iter.g, state.av, state.hat_γ) # always m_o is 1 larger than m_n
                
        gradient!(state.∇f_temp, iter.F[state.inds[ind.i]], state.z_M[ind.m_o]) # update the gradient - ind.i counts 123123123456...
        state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
        gradient!(state.∇f_temp, iter.F[state.inds[ind.i]], state.z_M[ind.m_n]) # update the gradient
        state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
        state.av .+= (state.hat_γ / state.γ[state.inds[ind.i]]) .* (state.z_M[ind.m_n] .- state.z_M[ind.m_o])
    end 

    return state, state
end

solution(state::FINITO_DFlbfgs_state) = state.z_M[end]
epoch_count(state::FINITO_DFlbfgs_state) = state.τ   # number of epochs is 2+ 1/tau , where 1/tau is from ls 




###### there is some bug when rep inseq =(1,1) it is not working......check after adding adaptive stepsizes!!!!