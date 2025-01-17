struct FINITO_lbfgs_adaptive_DNN_iterable{R<:Real,Tf,Tg, TH, Tf_full,Tp} <: CIAO_iterable
    F::Tf          # smooth term 
    F_full::Tf_full 
    g::Tg                  # nonsmooth term 
    opt_params::Tp                # initial point
    N::Int                    # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    η::R                    # ls division parameter for γ
    β::R                    # ls division parameter for τ
    sweeping::Int8          # to only use one stepsize γ
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    H::TH
    adaptive::Bool          # adaptive stepsizes
    tol_b::R
    data::Tuple
    DNN_config!
end

mutable struct FINITO_lbfgs_adaptive_DNN_state{R<:Real,Tx, TH}
    γ::Maybe{Tx}                   # stepsize parameter, plays the rolde of hat gamma  
    hat_γ::R                # average γ 
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches 
    H::TH                   # Hessian approx
    
    # some extra placeholders
    sum_z::Tx
    sum_fx::R
    sum_fv::R
    sum_nrmx::R             # sum |x_i|^2/γ_i
    sum_nabla::Tx           # sum \nabla f(x_i)
    sum_innprod::R          # sum  < \nabla f(x_i), x_i >           
    val_fg::R               # value of g        
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
    zbar::Tx                # bar z 
    zbar_prev::Maybe{Tx}    # bar z previous
    vbar::Tx                # bar v
    vbar_prev::Maybe{Tx}    # bar v previous
    dir::Tx                 # direction
    xbar::Tx                # linesearch candidate
    inds::Array{Int}        # needed for shuffled only! 
    τ::Float64              # number of epochs
end

function FINITO_lbfgs_adaptive_DNN_state(γ::Tx, hat_γ::R, av::Tx, ind, d, H::TH, sum_z, sum_nrmx, sum_nabla, sum_innprod, sum_fx) where {R,Tx,TH}
    return FINITO_lbfgs_adaptive_DNN_state{R,Tx,TH}(
        γ,
        hat_γ,
        av,
        ind,
        d,
        H, 
        sum_z,
        sum_fx,
        R(0),
        sum_nrmx,
        sum_nabla,
        sum_innprod,
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

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

function Base.iterate(iter::FINITO_lbfgs_adaptive_DNN_iterable{R}) where {R}
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
    # sum_nabla = zero(iter.x0)
    # sum_innprod = R(0)
    # sum_fx = R(0)
    # ∇f = 0
    # fi_x = 0
    # for i = 1:N  # nabla f(x0)
    #     # ∇f, fi_x = gradient(iter.F(i), iter.x0)
    #     ∇f = Flux.gradient(() -> iter.F(i,iter.x0), params(iter.x0))[iter.x0][:,1]
    #     fi_x = iter.F(i,iter.x0)
    #     sum_innprod += real(dot(∇f, iter.x0))
    #     sum_nabla .+= ∇f
    #     sum_fx += fi_x
    # end


    # sum_nabla = Flux.gradient(() -> iter.F_full(iter.opt_params), params(iter.x0))[iter.x0][:,1] * N
    # sum_fx = iter.F_full(iter.x0) * N
    # sum_innprod = real(dot(sum_nabla, iter.x0))

    gs = Flux.gradient(iter.opt_params) do # sampled gradient
        iter.F(batchmemaybe(iter.data)...)
    end
    sum_nabla = iter.DNN_config!(gs=gs) * N
    sum_fx = iter.F(batchmemaybe(iter.data)...) * N
    x0 = copy(iter.DNN_config!())
    sum_innprod = real(dot(sum_nabla, x0))

    # println("size testing ∇f: $(size(∇f))")
    # println("size testing fi_x: $(size(fi_x))")
    # println("size testing real.dot: $(size(real(dot(∇f, iter.x0))))")
    # println("size testing sum_nabla: $(size(sum_nabla))")
    # updating the stepsize 

    ##-------- γ initialization type 1-------------
    γ = iter.γ
    # xeps = x0 .+ one(R)
    #    av_eps = zero(x0)
    #    for i in 1:N
    #         # ∇f, ~ = gradient(iter.F[i], xeps)
    #         ∇f = Flux.gradient(() -> iter.F(i), params(iter.x0))
    #         av_eps .+= ∇f[iter.x0]
    #     end

    # iter.DNN_config!(gs=xeps)
    # gs = Flux.gradient(iter.opt_params) do # sampled gradient
    #     iter.F(batchmemaybe(iter.data)...)
    # end
    # av_eps = iter.DNN_config!(gs=gs) * N

    # nmg = norm(sum_nabla - av_eps)
    # t = 1
    # while nmg < eps(R)  # in case xeps has the same gradient
    #     println("initial upper bound for L is too small")
    #     xeps = x0 .+ rand(t * [-1, 1], size(x0))
    #     # av_eps = zero(x0)
    #     # for i in 1:N
    #     #     # ∇f, ~ = gradient(iter.F[i], xeps)
    #     #     ∇f = Flux.gradient(() -> iter.F(i), params(iter.x0))
    #     #     av_eps .+= ∇f[iter.x0]
    #     # end
    #     iter.DNN_config!(gs=xeps)
    #     gs = Flux.gradient(iter.opt_params) do # sampled gradient
    #         iter.F(batchmemaybe(iter.data)...)
    #     end
    #     av_eps = iter.DNN_config!(gs=gs) * N

    #     # grad_f_xeps, f_xeps = gradient(iter.F[i], xeps)
    #     nmg = norm(sum_nabla - av_eps)
    #     t *= 2
    # end
    ###### decide how to initialize! 
    # L_int = nmg / (t * sqrt(length(x0)))
    # L_int = 3.0 # no ls 
    # L_int /= iter.N # to account for 1/N
    # γ = iter.α / (L_int)
    # γ = iter.S(iter.x0)
    # println("gamma specified by L_int: ", γ); flush(stdout)

    ##-------- γ initialization type 2-------------
    # γ = 5.0
    # println("gamma specified by hand: ", γ); flush(stdout)
    ##---------------------------------------------

    γ = isa(γ, R) ? N * fill(γ, (N,)) : γ
    hat_γ = 1 / sum(1 ./ γ)
    # update the average
    av = copy(sum_nabla) .* (-hat_γ / N)
    av .+= x0
    # println("gamma: $(γ)"); flush(stdout)
    println("gamma hat: $(hat_γ)"); flush(stdout)
    println("size testing x0: $(size(x0))"); flush(stdout)

    sum_nrmx = norm(x0)^2 / hat_γ
    sum_z = x0 / hat_γ
    state = FINITO_lbfgs_adaptive_DNN_state(γ, hat_γ, av, ind, cld(N, r), iter.H, sum_z, sum_nrmx, sum_nabla, sum_innprod, sum_fx)

    println("test: $(norm(x0)) - $(hat_γ)")
    return state, state
end

function Base.iterate(
    iter::FINITO_lbfgs_adaptive_DNN_iterable{R},
    state::FINITO_lbfgs_adaptive_DNN_state{R},
) where {R}
    
    # println("outer iter started 1"); flush(stdout)
    
    if state.zbar_prev === nothing
        state.zbar_prev = zero(state.z)
        state.vbar_prev = zero(state.z)
    end

    γ_init = state.hat_γ

    # full z update 
     while true
        if state.hat_γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small at ls 1, ($(state.hat_γ))"
            return nothing
        end
        prox!(state.zbar, iter.g, state.av, state.hat_γ)

        # sum_fz = 0
        # for i in 1:iter.N 
        #     sum_fz += iter.F(i,state.zbar)
        # end 
        # sum_fz = iter.F_full(state.zbar) * iter.N
        iter.DNN_config!(gs=state.zbar)
        sum_fz = iter.F(batchmemaybe(iter.data)...) * iter.N


        ls_temp_l = sum_fz
        ls_temp_l -= real(dot(state.sum_nabla, state.zbar))
        ls_temp_l -= state.sum_fx 
        ls_temp_l += state.sum_innprod
        temp = (norm(state.zbar)^2)/(2*state.hat_γ) + (state.sum_nrmx)/2
        ls_temp_r =  (1 + 10^-6) * temp - real(dot(state.sum_z, state.zbar))
        ls_temp_r *= iter.N * iter.α

        tol = 10^(-6)  * (1 + abs(ls_temp_r))
        # break # no ls 
        R(ls_temp_l) <= ls_temp_r + tol && break
        println("ls 1"); flush(stdout)
        # println("x: ", ls_temp_l)
        # println("y: ", ls_temp_r)
        # println("hat_γ: ", state.hat_γ)
        # println("hat_calc_γ: ", 1 / sum(1 ./ state.γ))
        # println("norm: ", state.sum_nrmx)
        # println("p1: ", (norm(state.zbar)^2)/(2*state.hat_γ) + (state.sum_nrmx)/2)
        # println("p2: ", real(dot(state.sum_z, state.zbar)))


        γ_prev = state.hat_γ
        state.hat_γ *= iter.η
        # update av
        state.av .-=  ((state.hat_γ - γ_prev) / iter.N) .* state.sum_nabla
        reset!(state.H)
    end   

    # state.sum_fx = R(0)
    # state.sum_innprod = R(0)
    # state.sum_nabla = zero(state.av)
    # for i = 1:iter.N
    #     state.∇f_temp .= (Flux.gradient(() -> iter.F(i,state.zbar), params(state.zbar)))[state.zbar][:,1]
    #     fi_z = iter.F(i,state.zbar)
    #     # state.∇f_temp, fi_z = gradient(iter.F[i], state.zbar) # update the gradient
    #     state.sum_fx += fi_z 
    #     state.sum_nabla .+= state.∇f_temp 
    # end
    # state.sum_nabla .= (Flux.gradient(() -> iter.F_full(state.zbar), params(state.zbar)))[state.zbar][:,1] * iter.N
    # state.sum_fx = iter.F_full(state.zbar) * iter.N


    gs = Flux.gradient(iter.opt_params) do # sampled gradient
        iter.F(batchmemaybe(iter.data)...)
    end
    state.sum_nabla .= iter.DNN_config!(gs=gs) * iter.N
    state.sum_fx = iter.F(batchmemaybe(iter.data)...) * iter.N
    state.sum_innprod = real(dot(state.sum_nabla,state.zbar))
    state.av .= state.zbar .- (state.hat_γ / iter.N) .* state.sum_nabla
    state.sum_nrmx = norm(state.zbar)^2 * iter.N
    


    # println("outer iter started 2"); flush(stdout)

    # full v update 
    while true
        if state.hat_γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small at ls 2, ($(state.hat_γ))"
            return nothing
        end
        state.val_fg = prox!(state.vbar, iter.g, state.av, state.hat_γ) # bar v update
        # state.sum_fv = 0
        # for i in 1:iter.N 
        #     state.sum_fv += iter.F(i,state.vbar)
        # end
        # state.sum_fv = iter.F_full(state.vbar) * iter.N

        iter.DNN_config!(gs=state.vbar)
        state.sum_fv = iter.F(batchmemaybe(iter.data)...) * iter.N
        

        f_model_cnst = state.sum_nrmx / (2 * state.hat_γ) + state.sum_fx - state.sum_innprod   

        f_model_z = real(dot(state.vbar, state.av))
        f_model_z -=  norm(state.vbar)^2 / 2
        f_model_z *= iter.N / state.hat_γ

        f_model_z += state.sum_fv
        tol = 10^(-6)  * (1 + abs(f_model_cnst))
        # tol = eps(R)
        # break # no ls 
        R(f_model_z) <= f_model_cnst + tol && break
        println("ls 2")
        flush(stdout)

        γ_prev = state.hat_γ
        state.hat_γ *= iter.η
        # update av
        state.av .-=  ((state.hat_γ - γ_prev) / iter.N) .* state.sum_nabla
        reset!(state.H)
    end   
    state.vbar .-= state.zbar # \bar v- \bar z 
    
    # prepare for linesearch: compute varphi(bar z) 
    envVal = state.sum_fx / iter.N
    envVal += state.val_fg 
    envVal += real(dot(state.sum_nabla, state.vbar)) / iter.N
    envVal += norm(state.vbar)^2 / (2 *  state.hat_γ)

    # update lbfgs
    update!(state.H, state.zbar - state.zbar_prev, -state.vbar +  state.vbar_prev) 
    mul!(state.dir, state.H, state.vbar)
    # store vectors for next update
    copyto!(state.zbar_prev, state.zbar)
    copyto!(state.vbar_prev, state.vbar)
    # println("outer iter started 3"); flush(stdout)


    state.τ = 1
    γ_prev = state.hat_γ
    # while true  ### z_line should be removed using some other placeholder
    for i=1:5
        state.hat_γ = γ_prev
        state.xbar .=  state.zbar .+ (1- state.τ) .* state.vbar + state.τ * state.dir
        
        # compute varphi(xbar) 
        # state.sum_nabla = zero(state.temp)
        # state.sum_fx = 0
        # for i = 1:iter.N
        #     # state.∇f_temp, fi_z = gradient(iter.F[i], state.xbar) # update the gradient
        #     state.∇f_temp .= (Flux.gradient(() -> iter.F(i,state.xbar), params(state.xbar)))[state.xbar][:,1]
        #     fi_z = iter.F(i,state.xbar)
        #     state.sum_fx += fi_z / iter.N
        #     state.sum_nabla .+= state.∇f_temp 
        # end
        # state.sum_nabla .= (Flux.gradient(() -> iter.F_full(state.xbar), params(state.xbar)))[state.xbar][:,1] * iter.N
        # state.sum_fx = iter.F_full(state.xbar)

        iter.DNN_config!(gs=state.xbar)
        gs = Flux.gradient(iter.opt_params) do # sampled gradient
            iter.F(batchmemaybe(iter.data)...)
        end
        state.sum_nabla .= iter.DNN_config!(gs=gs) * iter.N
        state.sum_fx = iter.F(batchmemaybe(iter.data)...) 

        while true
            if state.hat_γ < iter.tol_b / iter.N
                @warn "parameter `γ` became too small at ls 3, ($(state.hat_γ))"
                return nothing
            end
            state.av .= state.xbar 
            state.av .-= (state.hat_γ / iter.N) .* state.sum_nabla
            state.val_fg = prox!(state.z, iter.g, state.av, state.hat_γ)
            
            # temp = 0
            # for i = 1:iter.N
            #     # ~, fi_z = gradient(iter.F[i], state.z) # update the gradient
            #     fi_z = iter.F(i,state.z)
            #     temp += fi_z / iter.N
            # end
            # temp = iter.F_full(state.z)
            iter.DNN_config!(gs=state.z)
            temp = iter.F(batchmemaybe(iter.data)...) 


            envVal_trial = 0
            envVal_trial += state.sum_fx
            state.z .-= state.xbar # r_γ(z^trial) 
            envVal_trial += real(dot(state.sum_nabla, state.z)) / iter.N
            envVal_trial += norm(state.z)^2 / (2 *  state.hat_γ)

            tol = 10^(-6)  * (1 + abs(envVal_trial))
            # tol = eps(R)
            # break # no ls 
            if temp <= envVal_trial + tol
                # println("ls condition met")
                break
            end
            state.hat_γ *= iter.η
            println("ls 3"); flush(stdout)
        end
        envVal_trial += state.val_fg

        tol = 10^(-6) * abs(envVal)
        envVal_trial <= envVal + tol && break
        state.τ *= iter.β      
        println("ls on τ"); flush(stdout)
        # println("envVal_trial: ", envVal_trial)
        # println("envVal      : ", envVal)
    end
    # println("outer iter started 4"); flush(stdout)

    state.zbar .= state.xbar # of line search
    ###### at the momoent I am computing the next prox twice (it is already computed in the ls) 
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled
    state.temp .= state.av
  
    state.sum_fx = R(0)
    state.sum_z = zero(state.temp)
    state.sum_innprod =  R(0)
    state.sum_nrmx =  R(0)
    while_flag = false
    flag = true

    state.γ *= (state.hat_γ / γ_init) 
    state.zbar .= state.xbar
    
    for j in state.inds
        for i in state.ind[j]
            # backtrack γ (warn if γ gets too small)  

            # println("γ_i: ", iter.S(state.zbar, i)/iter.N)
            #--------------------------- restarting section ---------------------------
            # hat_γ_prev = state.hat_γ
            # γ_prev = state.γ[i]
            # # gradient!(state.∇f_temp, iter.F[i], state.zbar)
            # # state.γ[i] = iter.S(state.zbar, state.∇f_temp, i) 
            # state.γ[i] = iter.S(state.zbar, i) 
            # state.hat_γ = 1/(1/state.hat_γ - 1/γ_prev + 1/state.γ[i])
            # # state.hat_γ = 1 / sum(1 ./ state.γ)
            # state.av .+= (hat_γ_prev - state.hat_γ) / iter.N * state.sum_nabla + (state.hat_γ/state.γ[i] - hat_γ_prev/γ_prev) * state.xbar
            #--------------------------- restarting section ---------------------------

            # print("current gamma: $(state.hat_γ) - current index: $(state.γ[i])")
            temp_x = iter.data[1][:,i]
            temp_y = iter.data[2][:,i]
            # if mod(i,100) == 0
            #     println("inner iter: $(i)")
            # end
            while true
                if state.hat_γ < iter.tol_b / iter.N
                    @warn "parameter `γ` became too small in ls 4 ($(state.hat_γ)) - for index $(state.γ[i])"
                    return nothing
                end
                prox!(state.zbar, iter.g, state.av, state.hat_γ)

                # fi_x = gradient!(state.∇f_temp, iter.F[i], state.xbar) # update the gradient
                # state.∇f_temp .= (Flux.gradient(() -> iter.F[i](state.xbar), params(state.xbar)))[state.xbar][:,1]
                # fi_x = iter.F[i](state.xbar)

                iter.DNN_config!(gs=state.xbar)
                gs = Flux.gradient(iter.opt_params) do # sampled gradient
                    iter.F(temp_x,temp_y)
                end
                state.∇f_temp .= iter.DNN_config!(gs=gs)
                fi_x = iter.F(temp_x,temp_y)
                
                # state.val_fg = iter.F[i](state.zbar)  ##### change to function value! #####
                iter.DNN_config!(gs=state.zbar)
                state.val_fg = iter.F(temp_x,temp_y)

                fi_model = fi_x + real(dot(state.∇f_temp, state.zbar .- state.xbar)) + 
                    (0.5 * iter.α * iter.N/ state.γ[i]) * (norm(state.zbar .- state.xbar)^2)

                tol = 10^(-6)  * (1 + abs(fi_model))
                # break # no ls 
                R(state.val_fg) <= fi_model + tol && break
                
                # println("ls 4 γ: $(state.hat_γ) for sample $(i)"); flush(stdout)
                # println("x: ", state.val_fg)
                # println("y: ", fi_model)

                hat_γ_prev = state.hat_γ
                γ_prev = state.γ[i]
                state.γ[i] *= iter.η # detach point!!
                # state.hat_γ = 1 / sum(1 ./ state.γ)
                state.hat_γ = 1/(1/state.hat_γ - 1/γ_prev + 1/state.γ[i]) # bug prone
                state.av .+= (hat_γ_prev - state.hat_γ) / iter.N * state.sum_nabla + (state.hat_γ/state.γ[i] - hat_γ_prev/γ_prev) * state.xbar

                reset!(state.H)
            end

            # iterates
            state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
            state.sum_nabla .-= state.∇f_temp # update sum nabla f_i for next iter    
            # gradient!(state.∇f_temp, iter.F[i], state.zbar) # update the gradient
            # state.∇f_temp .= (Flux.gradient(() -> iter.F[i](state.zbar), params(state.zbar)))[state.zbar][:,1]
            iter.DNN_config!(gs=state.zbar)
            gs = Flux.gradient(iter.opt_params) do # sampled gradient
                iter.F(temp_x,temp_y)
            end
            state.∇f_temp .= iter.DNN_config!(gs=gs)

            state.sum_nabla .+= state.∇f_temp # update sum nabla f_i for next iter
            state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
            state.av .+= state.hat_γ * (state.zbar .- state.xbar) ./ state.γ[i]

            # updates for the next linesearch
            state.sum_z += state.zbar / state.γ[i]
            state.sum_fx += state.val_fg
            state.sum_nrmx   += norm(state.zbar)^2 / state.γ[i]
            state.sum_innprod += real(dot(state.∇f_temp, state.zbar)) 
            # println("outer iter started 5"); flush(stdout)

        end
    end
    # println("current stepsize is $(state.hat_γ)")
    return state, state
end
solution(state::FINITO_lbfgs_adaptive_DNN_state) = state.zbar
epoch_count(state::FINITO_lbfgs_adaptive_DNN_state) = state.τ   # number of epochs is 2+ 1/tau , where 1/tau is from ls 

#### TODO: 
    # careful with the minibatch and adaptive stepsizes! 
    # stepsizes if gamma is given...
    # add LS parameter as iter field