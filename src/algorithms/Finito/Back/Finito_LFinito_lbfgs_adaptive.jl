struct FINITO_lbfgs_adaptive_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg, TH, Ts} <: CIAO_iterable
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                    # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    η::R                    # factor for updating gammas
    sweeping::Int8             # to only use one stepsize γ
    batch::Int                # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    H::TH
    adaptive::Bool          # adaptive stepsizes
    tol_b::R
    S::Ts            # functions s_i for calculating the Hassian of f_i s
end

mutable struct FINITO_lbfgs_adaptive_state{R<:Real,Tx, TH}
    γ::Tx                    # stepsize parameter
    hat_γ::R                # average γ 
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches 
    H::TH                   # Hessian approx
    # av_x::Tx                # sum x_i
    sum_fx::R
    sum_fv::R
    sum_x::Tx               # sum x_i/γ_i
    sum_nrmx::R               # sum |x_i|^2/γ_i
    sum_nabla::Tx            # sum \nabla f(x_i)
    sum_nabla_fz::Tx            # sum \nabla f(x_i)
    sum_innprod::R           # sum  < \nabla f(x_i), x_i >
    f_x::R                  # sum f(x_i) 
    # some extra placeholders           
    val_fg::R                   # value of g        
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
    zbar::Tx                # bar z 
    zbar_prev::Maybe{Tx}    # bar z previous
    vbar::Tx            # v
    vbar_prev::Maybe{Tx}# v
    dir::Tx                 # direction
    # sum_nabla::Tx              # for linesearch
    z_trial::Tx             # linesearch candidate
    inds::Array{Int}        # needed for shuffled only! 
    τ::Float64              # number of epochs
end

function FINITO_lbfgs_adaptive_state(γ::Tx, hat_γ::R, av::Tx, ind, d, H::TH, sum_x, sum_nrmx, sum_nabla, sum_innprod, f_x) where {R,Tx,TH}
    return FINITO_lbfgs_adaptive_state{R,Tx,TH}(
        γ,
        hat_γ,
        av,
        ind,
        d,
        H, 
        R(0),
        R(0),
        sum_x,
        sum_nrmx,
        sum_nabla,
        copy(av),
        sum_innprod,
        f_x,
        R(0),
        copy(av),
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
            # γ = iter.S(iter.x0)
            println("gamma specified by L_int: ", γ)
        else 
            println("gamma specified by L")
            isa(iter.L, R) ? (γ = iter.α * R(iter.N) / iter.L) :
                (γ = iter.α * R(N) / (maximum(iter.L)) )
        end
    else
        @warn "stepsize γ unspecified"
    end
    γ = fill(γ, (N,))
    hat_γ = 1 / sum(1 ./ γ)
    println("gggggg: ", hat_γ)
    # update the average
    av = copy(sum_nabla) .* (-hat_γ / N)
    av .+= iter.x0

    sum_x = iter.x0 /hat_γ
    sum_nrmx = N * norm(iter.x0)^2
    state = FINITO_lbfgs_adaptive_state(γ, hat_γ, av, ind, cld(N, r), iter.H, sum_x, sum_nrmx, sum_nabla, sum_innprod, f_x)

    return state, state
end

function Base.iterate(
    iter::FINITO_lbfgs_adaptive_iterable{R},
    state::FINITO_lbfgs_adaptive_state{R},
) where {R}
    
    if state.zbar_prev === nothing
        state.zbar_prev = zero(state.z)
        state.vbar_prev = zero(state.z)
    end

    # ------------- reset gamma by hessian initialization -------------
    # γ_prev = state.γ
    # state.γ = iter.S(state.z)
    # state.av .-=  ((state.γ - γ_prev) / iter.N) .* state.sum_nabla

    # full z update 
     while true
        if state.hat_γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small at ls 1, ($(state.γ))"
            return nothing
        end
        prox!(state.zbar, iter.g, state.av, state.hat_γ)

        f_model_cnst = iter.N * state.sum_nrmx / 2 + state.f_x - state.sum_innprod  
        
        f_model_z = iter.N * real(dot(state.zbar, state.sum_x))
        f_model_z -=  iter.N * norm(state.zbar)^2 / (2 * state.hat_γ)
        f_model_z -= real(dot(state.zbar, state.sum_nabla))
        for i in 1:iter.N 
            f_model_z += iter.F[i](state.zbar)
        end 

        tol = 10^(-6)  * (1 + abs(f_model_cnst))
        # tol = eps(R)
        R(f_model_z) <= f_model_cnst + tol && break
        println("ls 1")
        println("x: ", f_model_z)
        println("y: ", f_model_cnst)
        println("γ: ", state.γ)
        println("sum_nrmx: ", state.sum_nrmx)

        γ_prev = state.hat_γ
        state.γ *= iter.η
        state.hat_γ *= iter.η
        # updates
        state.av .-=  ((state.hat_γ - γ_prev) / iter.N) .* state.sum_nabla
        state.sum_nrmx ./= iter.η
        state.sum_x ./= iter.η
        reset!(state.H)
    end   

    state.f_x = R(0)
    state.sum_innprod = R(0)
    state.sum_nabla = zero(state.av)
    for i = 1:iter.N
        state.∇f_temp, fi_z = gradient(iter.F[i], state.zbar) # update the gradient
        state.f_x += fi_z 
        state.sum_nabla .+= state.∇f_temp 
    end
    state.sum_innprod += real(dot(state.sum_nabla,state.zbar))
    state.av .= state.zbar .- (state.hat_γ / iter.N) .* state.sum_nabla
    state.sum_nrmx = norm(state.zbar)^2 * iter.N
    state.sum_nabla_fz .= state.sum_nabla

    # full v update 
    while true
        if state.hat_γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small at ls 2, ($(state.γ))"
            return nothing
        end
        state.val_fg = prox!(state.vbar, iter.g, state.av, state.hat_γ) # bar v update
        state.sum_fv = 0
        for i in 1:iter.N 
            state.sum_fv += iter.F[i](state.vbar)
        end

        f_model_cnst = iter.N * norm(state.vbar - state.zbar)^2/(2 * state.hat_γ)

        f_model_z = - real(dot(state.sum_nabla, state.vbar - state.zbar))
        f_model_z += state.sum_fv 
        f_model_z -=  state.f_x

        tol = 10^(-6)  * (1 + abs(f_model_cnst))
        R(f_model_z) <= f_model_cnst + tol && break
        println("ls 2")

        γ_prev = state.hat_γ
        state.hat_γ *= 0.8
        state.γ *= 0.8
        # update av
        state.av .-=  ((state.hat_γ - γ_prev) / iter.N) .* state.sum_nabla_fz
        reset!(state.H)
    end   
    state.vbar .-= state.zbar # \bar v- \bar z 
    
    # prepare for linesearch: compute varphi(bar z) 
    envVal = state.f_x / iter.N
    envVal += state.val_fg 
    envVal += real(dot(state.sum_nabla, state.vbar)) / iter.N
    envVal += norm(state.vbar)^2 / (2 *  state.hat_γ)

    # update lbfgs
    update!(state.H, state.zbar - state.zbar_prev, -state.vbar +  state.vbar_prev) 
    mul!(state.dir, state.H, state.vbar)
    # store vectors for next update
    copyto!(state.zbar_prev, state.zbar)
    copyto!(state.vbar_prev, state.vbar)

    state.τ = 1.0
    hat_γ_prev = state.hat_γ
    γ_prev = state.γ
    # while true  ### z_line should be removed using some other placeholder
    for i=1:5
        state.hat_γ = hat_γ_prev
        state.γ = γ_prev
        state.z_trial .=  state.zbar .+ (1- state.τ) .* state.vbar + state.τ * state.dir
        
        # compute varphi(z_trial) 
        state.sum_nabla = zero(state.temp)
        state.sum_fx = 0
        for i = 1:iter.N
            state.∇f_temp, fi_z = gradient(iter.F[i], state.z_trial) # update the gradient
            state.sum_fx += fi_z 
            state.sum_nabla .+= state.∇f_temp 
        end

        while true
            if state.hat_γ < iter.tol_b / iter.N
                @warn "parameter `γ` became too small at ls 3, ($(state.γ))"
                return nothing
            end
            state.av .= state.z_trial 
            state.av .-= (state.hat_γ / iter.N) .* state.sum_nabla
            state.val_fg = prox!(state.z, iter.g, state.av, state.hat_γ)
            
            temp = 0
            for i = 1:iter.N
                ~, fi_z = gradient(iter.F[i], state.z) # update the gradient
                temp += fi_z 
            end

            envVal_trial = 0
            envVal_trial += state.sum_fx
            state.z .-= state.z_trial # r_γ(z^trial) 
            envVal_trial += real(dot(state.sum_nabla, state.z)) 
            envVal_trial += iter.N * norm(state.z)^2 / (2 *  state.hat_γ)

            tol = 10^(-6)  * (1 + abs(envVal_trial))
            if temp <= envVal_trial + tol
                break
            end
            state.hat_γ *= 0.8
            state.γ *= 0.8
            println("ls 3")
        end
        envVal_trial /= iter.N
        envVal_trial += state.val_fg

        envVal_trial <= envVal + eps(R) && break
        state.τ /= 50       ##### bug prone: change in reporting if you change this! ######
        println("ls on τ")
    end
    state.zbar .= state.z_trial # of line search
    ###### at the momoent I am computing the next prox twice (it is already computed in the ls) 
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled
  
    state.f_x = R(0)
    state.sum_innprod =  R(0)
    state.sum_nrmx =  R(0)
    state.sum_x = zero(state.temp)
    
    for j in state.inds
        for i in state.ind[j]
            # backtrack γ (warn if γ gets too small)  

            hat_γ_prev = state.hat_γ
            γ_prev = state.γ[i]
            state.γ[i] = iter.S(state.zbar, i)
            
            while true
                state.hat_γ = 1/sum(1/state.γ)

                if state.hat_γ < iter.tol_b / iter.N
                    @warn "parameter `γ` became too small ($(state.γ))"
                    return nothing
                end
                
                println("norm av: ", norm(state.av))
                state.av .= state.av + (hat_γ_prev - state.hat_γ)/iter.N .* state.sum_nabla + (state.hat_γ/state.γ[i] - hat_γ_prev/γ_prev) * state.zbar
                prox!(state.z, iter.g, state.av, state.hat_γ)

                fi_x = gradient!(state.∇f_temp, iter.F[i], state.zbar) # update the gradient
                state.val_fg = iter.F[i](state.z)  ##### change to function value! #####
                state.temp .= state.z .- state.zbar # placeholder for z - x^full

                fi_model = fi_x + real(dot(state.∇f_temp, state.temp)) + 
                    (0.5 * iter.α * iter.N / state.γ[i]) * (norm(state.temp)^2)

                tol = 10^(-6)  * (1 + abs(fi_model))
                R(state.val_fg) <= fi_model + tol && break
                
                println("ls 4: ", state.hat_γ)
                println("x: ", state.val_fg)
                println("y: ", fi_model)
                println("sum_nabla: ", norm(state.sum_nabla))
                println("norm: ", state.γ[i])
                println("norm: ", norm(state.av))
                println("norm: ", norm(state.zbar))
                println("av+: ", norm((hat_γ_prev - state.hat_γ)/iter.N .* state.sum_nabla + (state.hat_γ/state.γ[i] - hat_γ_prev/γ_prev) * state.zbar))
                println("dot: ", (state.hat_γ/state.γ[i] - hat_γ_prev/γ_prev))
                println("dot: ", (hat_γ_prev - state.hat_γ)/iter.N)
                return nothing
                γ_prev = state.γ[i]
                hat_γ_prev = state.hat_γ

                state.γ[i] *= iter.η
                reset!(state.H)
            end

            # iterates
            state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
            state.sum_nabla .-= state.∇f_temp # update sum nabla f_i for next iter    
            gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
            state.sum_nabla .+= state.∇f_temp # update sum nabla f_i for next iter
            state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
            state.av .+= state.hat_γ * (state.z .- state.zbar) ./ state.γ[i]

            # updates for the next linesearch
            state.f_x += state.val_fg
            state.sum_x += state.z / state.γ[i]
            state.sum_nrmx   += norm(state.z)^2 /state.γ[i]
            state.sum_innprod += real(dot(state.∇f_temp, state.z)) 
        end
    end
    # println("current stepsize is $(state.γ)")
    return state, state
end
solution(state::FINITO_lbfgs_adaptive_state) = state.z
epoch_count(state::FINITO_lbfgs_adaptive_state) = state.τ   # number of epochs is 2+ 1/tau , where 1/tau is from ls 

#### TODO: 
    # careful with the minibatch and adaptive stepsizes! 
    # stepsizes if gamma is given...
    # add LS parameter as iter field