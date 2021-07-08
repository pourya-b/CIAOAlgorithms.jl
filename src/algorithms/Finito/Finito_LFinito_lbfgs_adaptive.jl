struct FINITO_lbfgs_adaptive_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg, TH} <: CIAO_iterable
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
    adaptive::Bool              # adaptive stepsizes
    tol_b::R
end

mutable struct FINITO_lbfgs_adaptive_state{R<:Real,Tx, TH}
    γ::R                    # stepsize parameter, plays the rolde of hat gamma  
    # γ::R                # average γ 
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches 
    H::TH                   # Hessian approx
    # av_x::Tx                # sum x_i
    sum_nrmx::R               # sum |x_i|^2
    sum_nabla::Tx            # sum \nabla f(x_i)
    sum_innprod::R           # sum  < \nabla f(x_i), x_i >
    f_x::R                  # sum f(x_i) 
    # some extra placeholders           
    val_fg::R                   # value of g        
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    zbar::Tx                # bar z 
    zbar_prev::Maybe{Tx}    # bar z previous
    res_zbar::Tx            # v
    res_zbar_prev::Maybe{Tx}# v
    dir::Tx                 # direction
    # sum_nabla::Tx              # for linesearch
    z_trial::Tx             # linesearch candidate
    inds::Array{Int}        # needed for shuffled only! 
    τ::Float64              # number of epochs
end

function FINITO_lbfgs_adaptive_state(γ::R, av::Tx, ind, d, H::TH, sum_nrmx, sum_nabla, sum_innprod, f_x) where {R,Tx,TH}
    return FINITO_lbfgs_adaptive_state{R,Tx,TH}(
        γ,
        av,
        ind,
        d,
        H, 
        sum_nrmx,
        sum_nabla,
        sum_innprod,
        f_x,
        R(0),
        copy(av),
        copy(av),
        copy(av),
        nothing,
        copy(av),
        nothing,
        copy(av),
        copy(av),
        collect(1:d),
        1.0
        )
end

function Base.iterate(iter::FINITO_lbfgs_adaptive_iterable{R}) where {R}
    N = iter.N
    r = iter.batch # batch size 
    # create index sets 
    ind = Vector{Vector{Int}}(undef, 0)
    d = Int(floor(N / r))
    for i = 1:d
        push!(ind, collect(r*(i-1)+1:i*r))
    end
    r * d < N && push!(ind, collect(r*d+1:N))

    #initializing the vectors 
    sum_nabla = zero(iter.x0)
    sum_innprod = R(0)
    f_x = R(0)
    for i = 1:N  # nabla f(x0)
        ∇f, fi_x = gradient(iter.F[i], iter.x0)
        sum_innprod += real(dot(∇f, iter.x0))
        sum_nabla .+= ∇f
        f_x += fi_x
    end

    # updating the stepsize 
    if iter.γ === nothing ##  Lip. of f(x) = 1/N sum nabla f_i(x) 
        if iter.L === nothing ||  iter.adaptive
           xeps = iter.x0 .+ one(R)
           av_eps = zero(iter.x0)
           for i in 1:N
                ∇f, ~ = gradient(iter.F[i], xeps)
                av_eps .+= ∇f
            end
            nmg = norm(sum_nabla - av_eps)
            t = 1
            while nmg < eps(R)  # in case xeps has the same gradient
                println("initial upper bound for L is too small")
                xeps = iter.x0 .+ rand(t * [-1, 1], size(iter.x0))
                av_eps = zero(iter.x0)
                for i in 1:N
                    ∇f, ~ = gradient(iter.F[i], xeps)
                    av_eps .+= ∇f
                end
                # grad_f_xeps, f_xeps = gradient(iter.F[i], xeps)
                nmg = norm(sum_nabla - av_eps)
                t *= 2
            end
            ###### decide how to initialize! 
            L_int = nmg / (t * sqrt(length(iter.x0)))
            L_int /= iter.N # to account for 1/N
            γ = iter.α / (L_int)

        else 
            isa(iter.L, R) ? (γ = iter.α * R(iter.N) / iter.L) :
                (γ = iter.α * R(N) / (maximum(iter.L)) )
        end
    else
        @warn "stepsize γ unspecified"
    end
    # update the average
    av = copy(sum_nabla) .* (-γ / N)
    av .+= iter.x0

    sum_nrmx = N * norm(iter.x0)^2
        
    state = FINITO_lbfgs_adaptive_state(γ, av, ind, cld(N, r), iter.H, sum_nrmx, sum_nabla, sum_innprod, f_x)

    return state, state
end

function Base.iterate(
    iter::FINITO_lbfgs_adaptive_iterable{R},
    state::FINITO_lbfgs_adaptive_state{R},
) where {R}
    
    if state.zbar_prev === nothing
        state.zbar_prev = zero(state.z)
        state.res_zbar_prev = zero(state.z)
    end

    # full z update 
     while true
        if state.γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small ($(state.γ))"
            return nothing
        end
        prox!(state.zbar, iter.g, state.av, state.γ)

        f_model_cnst = state.sum_nrmx / (2 * state.γ) + state.f_x - state.sum_innprod   

        f_model_z = real(dot(state.zbar, state.av ))
        f_model_z -=  norm(state.zbar)^2 / 2
        # println("---------1------")

        f_model_z *= iter.N / state.γ
        for i in 1:iter.N 
            f_model_z += iter.F[i](state.zbar)
        end 
        
        tol = eps(R) # * (1 + abs(fi_z))
        R(f_model_z) <= f_model_cnst + tol && break

        γ_prev = state.γ
        state.γ *= 0.8
        # update av
        state.av .-=  ((state.γ - γ_prev) / iter.N) .* state.sum_nabla
        reset!(state.H)
    end   

    state.f_x = R(0)
    state.sum_innprod = R(0)
    # state.av .= state.zbar
    state.sum_nabla = zero(state.av)
    for i = 1:iter.N
        state.∇f_temp, fi_z = gradient(iter.F[i], state.zbar) # update the gradient
        state.f_x += fi_z 
        state.sum_nabla .+= state.∇f_temp 
    end
    state.sum_innprod += real(dot(state.sum_nabla,state.zbar))

    state.av .= state.zbar .- (state.γ / iter.N) .* state.sum_nabla

    state.sum_nrmx = norm(state.zbar)^2 * iter.N
    # full v update 
    while true
        if state.γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small ($(state.γ))"
            return nothing
        end
        state.val_fg = prox!(state.res_zbar, iter.g, state.av, state.γ) # bar v update

        f_model_cnst = state.sum_nrmx / (2 * state.γ) + state.f_x - state.sum_innprod   

        f_model_z = real(dot(state.res_zbar, state.av ))
        f_model_z -=  norm(state.res_zbar)^2 / 2
        f_model_z *= iter.N / state.γ
        for i in 1:iter.N 
            f_model_z += iter.F[i](state.res_zbar)
        end 
        
        tol = eps(R) # * (1 + abs(fi_z))
        R(f_model_z) <= f_model_cnst + tol && break

        γ_prev = state.γ
        state.γ *= 0.8
        # update av
        state.av .-=  ((state.γ - γ_prev) / iter.N) .* state.sum_nabla
        reset!(state.H)
    end   
    state.res_zbar .-= state.zbar # \bar v- \bar z 
    
    # prepare for linesearch: compute varphi(bar z) 
    envVal = state.f_x / iter.N
    envVal += state.val_fg 
    envVal += real(dot(state.sum_nabla, state.res_zbar)) / iter.N
    envVal += norm(state.res_zbar)^2 / (2 *  state.γ)

    # update lbfgs
    update!(state.H, state.zbar - state.zbar_prev, -state.res_zbar +  state.res_zbar_prev) 
    # store vectors for next update
    copyto!(state.zbar_prev, state.zbar)
    copyto!(state.res_zbar_prev, state.res_zbar)

    mul!(state.dir, state.H, state.res_zbar)

    state.τ = 1.0
    # while true  ### z_line should be removed using some other placeholder
    for i=1:5
        state.z_trial .=  state.zbar .+ (1- state.τ) .* state.res_zbar + state.τ * state.dir

        # compute varphi(z_trial) 
        state.sum_nabla = zero(state.av)
        envVal_trial = 0
        for i = 1:iter.N
            state.∇f_temp, fi_z = gradient(iter.F[i], state.z_trial) # update the gradient
            envVal_trial += fi_z / iter.N
            state.sum_nabla .+= state.∇f_temp 
        end
        state.av .= state.z_trial 
        state.av .-= (state.γ / iter.N) .* state.sum_nabla

        # prox( s(z^trial) )
        state.val_fg = prox!(state.z,iter.g, state.av, state.γ)
        envVal_trial += state.val_fg
        state.z .-= state.z_trial # r_γ(z^trial) 

        envVal_trial += real(dot(state.sum_nabla, state.z)) / iter.N
        envVal_trial += norm(state.z)^2 / (2 *  state.γ)

        envVal_trial <= envVal + eps(R) && break
        state.τ /= 50       ##### bug prone: change in reporting if you change this! ######
    end
    state.zbar .= state.z_trial # of line search

    ###### at the momoent I am computing the next prox twice (it is already computed in the ls) 
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled

    state.f_x = R(0)
    state.sum_innprod =  R(0)
    state.sum_nrmx =  R(0)
    
    for j in state.inds
        for i in state.ind[j]
        
            # backtrack γ (warn if γ gets too small)   
            while true
                if state.γ < iter.tol_b / iter.N
                    @warn "parameter `γ` became too small ($(state.γ))"
                    return nothing
                end
                prox!(state.z, iter.g, state.av, state.γ)

                fi_x = gradient!(state.∇f_temp, iter.F[i], state.zbar) # update the gradient

                state.val_fg = iter.F[i](state.z)  ##### change to function value! #####
                
                state.z_trial .= state.z .- state.zbar # placeholder for z - x^full

                fi_model = fi_x + real(dot(state.∇f_temp, state.z_trial)) + 
                    (0.5 * iter.α / state.γ) * (norm(state.z_trial)^2)
                tol = eps(R) * (1 + abs(state.val_fg))
                R(state.val_fg) <= fi_model + tol && break

                γ_prev = state.γ
                state.γ *= 0.8
                # update av
                state.av .-=  ((state.γ - γ_prev) / iter.N) .* state.sum_nabla

                reset!(state.H)
            end

            # iterates
            state.av .+= (state.γ / iter.N) .* state.∇f_temp
            state.sum_nabla -= state.∇f_temp # update sum nabla f_i for next iter    
            gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
            state.sum_nabla += state.∇f_temp # update sum nabla f_i for next iter
            state.av .-= (state.γ / iter.N) .* state.∇f_temp
            state.av .+= (state.z .- state.zbar) ./ iter.N

            # updates for the next linesearch
            state.f_x += state.val_fg
            state.sum_nrmx   += norm(state.z)^2 
            state.sum_innprod += real(dot(state.∇f_temp, state.z)) 
        end
    end

    println("current stepsize is $(state.γ)")

    return state, state
end

solution(state::FINITO_lbfgs_adaptive_state) = state.z


epoch_count(state::FINITO_lbfgs_adaptive_state) = state.τ   # number of epochs is 2+ 1/tau , where 1/tau is from ls 


#### TODO: 
    # careful with the minibatch and adaptive stepsizes! 
    # stepsizes if gamma is given...
    # add LS parameter as iter field