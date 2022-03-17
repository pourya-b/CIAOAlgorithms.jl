# adaSPIRAL 

struct FINITO_lbfgs_adaptive_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg, TH} <: CIAO_iterable
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    η::R                    # ls parameter for γ
    β::R                    # ls parameter for τ
    sweeping::Int8          # sweeping strategy in the inner loop, # 1:cyclical, 2:shuffled
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    H::TH                   # LBFGS struct
    tol_b::R                # γ backtracking stopping criterion
    ls_tol::R               # tolerance in ls
    D::Maybe{R}             # a large number to normalize lbfgs direction if provided
end

mutable struct FINITO_lbfgs_adaptive_state{R<:Real,Tx, TH, Mx}
    γ::Array{R}             # stepsize parameter
    hat_γ::R                # average γ (cf. Alg 2)
    s::Tx                   # the running average (vector s)
    ts::Tx                  # the running average (vector \tilde s)
    bs::Tx                  # the running average (vector \bar s)
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches 
    H::TH                   # Hessian approx
    
    # some extra placeholders
    sum_tz::Tx              # \sum{(tz^k_i)}
    sum_fz::R #10           # \sum f_i(z^k)
    sum_nrm_tz::R           # \sum |tz^k_i|^2/γ_i
    sum_∇fz::Tx             # \sum \nabla f_i(z)        
    sum_innprod_tz::R       # \sum  < \nabla f_i(tz^k_i), tz^k_i >           
    val_gv::R               # g(v^k)  
    val_gy::R               # g(y^k)
    sum_ftz::R              # \sum f_i(tz^k_i)
    y::Tx                   # y^k
    tz::Tx                  # \tilde z^k
    ∇f_temp::Tx             # placeholder for gradients 
    sum_∇fu::Tx #20         # \sum \nabla f_i(u^k)
    z::Tx                   # z^k
    z_prev::Maybe{Tx}       # previous z 
    v::Tx                   # v^k
    v_prev::Maybe{Tx}       # previous v
    dir::Tx                 # quasi_Newton direction d^k
    u::Tx                   # linesearch candidate u^k
    inds::Array{Int}        # An array containing the indices of block batches
    τ::Float64              # interpolation parameter between the quasi-Newton direction and the nominal step 
    ls_grad_eval::Int       # number of grad evaluations in the backtrackings of the stepsize (in ls 3)
    A::Mx
end

function FINITO_lbfgs_adaptive_state(γ, hat_γ::R, s0::Tx, ind, d, H::TH, sum_tz, sum_nrm_tz, sum_∇fz, sum_innprod_tz, sum_ftz, A::Mx) where {R,Tx,TH,Mx}
    return FINITO_lbfgs_adaptive_state{R,Tx,TH,Mx}(
        γ,
        hat_γ,
        s0,
        s0,
        s0,
        ind,
        d,
        H, 
        sum_tz,
        R(0), #10
        sum_nrm_tz,
        sum_∇fz,
        sum_innprod_tz,
        R(0),
        R(0),
        sum_ftz,
        copy(s0), 
        copy(s0), 
        copy(s0),
        sum_∇fz, #20
        copy(s0),
        nothing, # z_prev
        copy(s0),
        nothing,
        copy(s0),
        copy(s0),
        collect(1:d),
        1.0,
        0,
        A,
        )
end

function Base.iterate(iter::FINITO_lbfgs_adaptive_iterable{R}) where {R}
    N = iter.N
    n = size(iter.x0)[1]
    r = iter.batch # batch size 
    # create index sets 
    ind = Vector{Vector{Int}}(undef, 0)
    d = Int(floor(N / r))
    for i = 1:d
        push!(ind, collect(r*(i-1)+1:i*r))
    end
    r * d < N && push!(ind, collect(r*d+1:N))

    #initializing the vectors 
    sum_∇fz = zero(iter.x0)
    sum_innprod_tz = R(0) # as the initialization of sum_innprod_tz
    sum_ftz = R(0)
    for i = 1:N  # nabla f(x0)
        ∇f, fi_z = gradient(iter.F[i], iter.x0)
        sum_innprod_tz += real(dot(∇f, iter.x0))
        sum_∇fz .+= ∇f
        sum_ftz += fi_z  # as the initialization of sum_ftz
    end

    # stepsize initialization
    if iter.γ === nothing 
        if iter.L === nothing
           xeps = iter.x0 .+ one(R)
           av_eps = zero(iter.x0)
           for i in 1:N
                ∇f, ~ = gradient(iter.F[i], xeps)
                av_eps .+= ∇f
            end
            nmg = norm(sum_∇fz - av_eps)
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
                nmg = norm(sum_∇fz - av_eps)
                t *= 2
            end
            L_int = nmg / (t * sqrt(length(iter.x0)))
            γ = iter.N * iter.α / (L_int)
            γ = fill(γ, (N,)) # to make it a vector
            println("γ specified by L_int")
        else 
            γ = 10 * iter.α * R(iter.N) / minimum(iter.L)
            γ = fill(γ, (N,)) # to make it a vector
            println("gamma specified by min{L}/10")
        end
    else
        γ_max = maximum(iter.γ)
        γ = fill(γ_max, (N,)) # provided γ
        println("gamma specified by max{γ}")
    end
    #initializing the vectors 
    hat_γ = 1 / sum(1 ./ γ)
    s0 = copy(sum_∇fz) .* (-hat_γ / N)
    s0 .+= iter.x0

    sum_nrm_tz = norm(iter.x0)^2 / hat_γ  # as the initialization of sum_nrm_tz
    sum_tz = iter.x0 / hat_γ # as the initialization of sum_tz
    A = zeros(N,n)
    state = FINITO_lbfgs_adaptive_state(γ, hat_γ, s0, ind, cld(N, r), iter.H, sum_tz, sum_nrm_tz, sum_∇fz, sum_innprod_tz, sum_ftz,A)

    return state, state
end

function Base.iterate(
    iter::FINITO_lbfgs_adaptive_iterable{R},
    state::FINITO_lbfgs_adaptive_state{R},
) where {R}
    
    if state.z_prev === nothing  # for lbfgs updates
        state.z_prev = zero(iter.x0)
        state.v_prev = zero(iter.x0)
    end

    γ_init = state.hat_γ # to update the individual stepsizes before entering the inner loop, by the change happened to hat_γ over the first 3 ls.

    ls_lhs = 0
    ls_rhs = 0
    while true # first ls
        if state.hat_γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small at ls 1, ($(state.hat_γ))"
            return nothing
        end
        prox!(state.z, iter.g, state.s, state.hat_γ) # z^k in Alg. 1&2

        state.sum_fz = 0 # for ls 1
        for i in 1:iter.N 
            state.sum_fz += iter.F[i](state.z)
        end 

        ls_lhs = state.sum_fz #* # lhs and rhs of the ls condition (Table 1, 1b)
        ls_lhs -= real(dot(state.sum_∇fu, state.z))
        ls_lhs -= state.sum_ftz 
        ls_lhs += state.sum_innprod_tz

        temp =  (norm(state.z)^2)/(2*state.hat_γ) + (state.sum_nrm_tz)/2 
        ls_rhs = (1 + iter.ls_tol) * temp - real(dot(state.sum_tz, state.z)) # bug prone
        ls_rhs *= iter.N * iter.α       
        
        # tol = 10^(-6)  * (1 + abs(ls_rhs)) # bug prone
        # R(ls_lhs) <= ls_rhs + tol && break  # the ls condition (Table 1, 1b)
        ls_lhs <= ls_rhs + iter.ls_tol  && break  # the ls condition (Table 1, 1b)
        println("ls 1")
        
        sum_ftz = 0
        sum_∇ftz = zero(iter.x0)
        sum_innprod_tz = 0
        sum_nrm_tz = 0
        sum_tz = zero(iter.x0)
        for i in 1:iter.N 
            sum_ftz += iter.F[i](state.A[i,:])
            gradient!(state.∇f_temp, iter.F[i], state.A[i,:])
            sum_∇ftz += state.∇f_temp
            sum_innprod_tz += real(dot(state.∇f_temp, state.A[i,:]))
            sum_nrm_tz += norm(state.A[i,:])^2 / state.γ[i]
            sum_tz += state.A[i,:]/state.γ[i]
        end 

        ls_rhs_ =  (norm(state.z)^2)/(2*state.hat_γ) + (sum_nrm_tz)/2 - real(dot(sum_tz, state.z)) # bug prone
        ls_rhs_ *= iter.N * iter.α  

        ls_lhs_ = state.sum_fz #* # lhs and rhs of the ls condition (Table 1, 1b)
        ls_lhs_ -= real(dot(sum_∇ftz, state.z))
        ls_lhs_ -= sum_ftz 
        ls_lhs_ += sum_innprod_tz

        println("lhs : $(ls_lhs)  /  rhs : $(ls_rhs)")
        println("lhs_: $(ls_lhs_)  /  rhs_: $(ls_rhs_)")
        println("check: (1) $(sum_nrm_tz)/$(state.sum_nrm_tz - sum_nrm_tz) |  (2) $(real(dot(state.sum_tz, state.z))-real(dot(sum_tz, state.z)))")
        println("check: (3) $(real(dot(state.sum_∇fu, state.z)) - real(dot(sum_∇ftz, state.z))) |  (4) $(state.sum_innprod_tz-sum_innprod_tz) |  (5) $(state.sum_ftz - sum_ftz)")

        
        γ_prev = state.hat_γ
        state.hat_γ *= iter.η
        state.γ *= iter.η

        # update s^k
        state.s .-=  ((state.hat_γ - γ_prev) / iter.N) .* state.sum_∇fu
        state.sum_nrm_tz /= iter.η
        state.sum_tz ./= iter.η
        reset!(state.H)
        state.z_prev .= zero(state.s)
        state.v_prev .= zero(state.s)
    end   
    # println("z norm: $(norm(state.z))")
    # if norm(state.z) == 0
    #     println("s norm: $(norm(state.s))")
    #     println("hat_γ: $(norm(state.hat_γ))")
    #     return nothing
    # end

    # now z^k is fixed, moving to step 2

    sum_innprod_z = R(0) # for ls 2
    state.sum_∇fz = zero(state.s) # for ls 2 and \bar s^k
    for i = 1:iter.N    # full update
        gradient!(state.∇f_temp, iter.F[i], state.z)  
        state.sum_∇fz .+= state.∇f_temp 
    end
    sum_innprod_z += real(dot(state.sum_∇fz,state.z)) # for ls 2
    state.bs .= state.z .- (state.hat_γ / iter.N) .* state.sum_∇fz # \bar s^k
    nrmz = norm(state.z)^2 * iter.N # for ls 2

    while true # second ls
        if state.hat_γ < iter.tol_b / iter.N
            @warn "parameter `γ` became too small at ls 2, ($(state.hat_γ))"
            return nothing
        end
        state.val_gv = prox!(state.v, iter.g, state.bs, state.hat_γ) # v^k and g(v)
        sum_fv = 0 # for ls 2
        for i in 1:iter.N 
            sum_fv += iter.F[i](state.v)
        end

        ls_rhs = iter.N * norm(state.v - state.z)^2/(2*state.hat_γ)  # for ls 2 

        ls_lhs = real(dot(state.sum_∇fz, state.z - state.v)) # for ls 2
        ls_lhs += sum_fv
        ls_lhs -= state.sum_fz

        # tol = 10^(-6)  * (1 + abs(ls_rhs)) # bug prone!
        # R(ls_lhs) <= ls_rhs + tol && break  # the ls condition (Table 1, 3b)
        ls_lhs <= ls_rhs + iter.ls_tol && break  # the ls condition (Table 1, 3b)
        println("ls 2")

        γ_prev = state.hat_γ
        state.hat_γ *= iter.η
        state.γ *= iter.η

        # update \bar s^k
        state.bs .-=  ((state.hat_γ - γ_prev) / iter.N) .* state.sum_∇fz
        reset!(state.H)
        state.z_prev .= zero(state.s)
        state.v_prev .= zero(state.s)
    end   
    # now v^k and bs are fixed, moving to step 4
    
    # prepare for linesearch on τ
    envVal = state.sum_fz / iter.N  # envelope value (lyapunov function) L(v^k,z^k)
    envVal += state.val_gv 
    envVal += real(dot(state.sum_∇fz, state.v - state.z)) / iter.N
    envVal += norm(state.v-state.z)^2 / (2 *  state.hat_γ)

    # update lbfgs
    update!(state.H, state.z - state.z_prev, state.z-state.v +  state.v_prev) 
    # store vectors for next update

    mul!(state.dir, state.H, state.v-state.z) # updating the quasi-Newton direction
    
    if iter.D != nothing
        state.dir .= state.dir * D * norm(state.v-state.z)/norm(state.dir)
    end

    print("norm d: $(norm(state.dir))\n | y_s: $(norm(state.z-state.v +  state.v_prev)) / x_s: $(norm(state.z - state.z_prev))")
    copyto!(state.z_prev, state.z)
    copyto!(state.v_prev, state.v-state.z)

    state.τ = 1
    for i=1:5 # backtracking on τ
        while true # third ls
            if state.hat_γ < iter.tol_b / iter.N
                @warn "parameter `γ` became too small at ls 3, ($(state.hat_γ))"
                return nothing
            end

            state.u .=  state.z .+ (1- state.τ) .* (state.v-state.z) + state.τ * state.dir # u^k
            
            state.sum_∇fu = zero(state.sum_∇fu) # here state.sum_∇fu is sum of nablas
            sum_fu = 0
            for i = 1:iter.N # full update for \tilde s^k
                state.∇f_temp, fi_u = gradient(iter.F[i], state.u) 
                sum_fu += fi_u 
                state.sum_∇fu .+= state.∇f_temp 
            end

            state.ts .= state.u # \tilde s^k
            state.ts .-= (state.hat_γ / iter.N) .* state.sum_∇fu
            state.val_gy = prox!(state.y, iter.g, state.ts, state.hat_γ) # y^k
            
            sum_fy = 0
            for i = 1:iter.N # for the ls condition
                ~, fi_y = gradient(iter.F[i], state.y) 
                sum_fy += fi_y 
            end

            envVal_trial = 0 # for the ls condition
            envVal_trial += sum_fu / iter.N
            # state.y .-= state.u # 
            envVal_trial += real(dot(state.sum_∇fu, state.y-state.u)) / iter.N
            envVal_trial += norm(state.y-state.u)^2 / (2 *  state.hat_γ)

            ls_rhs = iter.N * norm(state.y - state.u)^2/(2*state.hat_γ)  # for ls 3
            ls_lhs = real(dot(state.sum_∇fu, state.u - state.y)) # for ls 3
            ls_lhs += sum_fy 
            ls_lhs -= sum_fu 

            # tol = 10^(-6)  * (1 + abs(envVal_trial)) # bug prone!
            # ls_lhs <= ls_rhs + tol && break  # the ls condition (Table 1, 5d)
            ls_lhs <= ls_rhs + iter.ls_tol && break  # the ls condition (Table 1, 5d)
            println("ls 3")

            state.ls_grad_eval += 1

            reset!(state.H)
            state.z_prev .= zero(state.s)
            state.v_prev .= zero(state.s)
            state.τ = 1
            γ_prev = state.hat_γ
            state.hat_γ *= iter.η # updating stepsize
            state.γ *= iter.η
            state.bs .-=  ((state.hat_γ - γ_prev) / iter.N) .* state.sum_∇fz # updating \bar s
            state.val_gv = prox!(state.v, iter.g, state.bs, state.hat_γ) # updating v

            # update lbfgs and the  direction (bug prone!)
            update!(state.H, state.z - state.z_prev, state.z-state.v +  state.v_prev) 
            copyto!(state.z_prev, state.z)
            copyto!(state.v_prev, state.v-state.z)
            mul!(state.dir, state.H, state.v-state.z) # updating the quasi-Newton direction
            if iter.D != nothing
                state.dir .= state.dir * D * norm(state.v-state.z)/norm(state.dir)
            end
            # print("norm d: $(norm(state.dir))")

            # updating the lyapunov function L(v^k,z^k)
            envVal = state.sum_fz / iter.N  
            envVal += state.val_gv 
            envVal += real(dot(state.sum_∇fz, state.v-state.z)) / iter.N
            envVal += norm(state.v-state.z)^2 / (2 *  state.hat_γ)
        end
        envVal_trial += state.val_gy  # envelope value (lyapunov function) L(y^k,u^k)

        tol = 10^(-8) * (1+abs(envVal)) # bug prone!
        # envVal_trial <= envVal + tol && break # descent on the envelope function (Table 1, 5e)
        # envVal_trial <= envVal + iter.ls_tol && break # descent on the envelope function (Table 1, 5e)
        envVal_trial <= envVal + eps(R) && break # descent on the envelope function (Table 1, 5e)
        state.τ *= iter.β   # backtracking on τ
        println("ls on τ")
    end

    # println("u norm: $(norm(state.u))")
    # if norm(state.u) == 0
    #     println("z norm: $(norm(state.z))")
    #     println("v norm: $(norm(state.v))")
    #     println("τ: $(norm(state.τ))")
    #     return nothing
    # end

    state.s .= state.ts # step 6

    iter.sweeping == 2 && (state.inds = randperm(state.d)) # shuffled (only shuffeling the batch indices not the indices inside of each batch)
  
    state.sum_ftz = R(0)
    state.sum_tz .= zero(state.ts)
    state.sum_innprod_tz =  R(0)
    state.sum_nrm_tz = R(0)
    state.A = zeros(iter.N,size(iter.x0)[1])

    # state.γ *= (state.hat_γ / γ_init) # updating individual stepsizes before entering the inner-loop
    
    for j in state.inds # batch indices
        for i in state.ind[j] # in Algorithm 1 line 7, batchsize is 1, but here we are more general - state.ind: indices in the j th batch
            
            while true # forth ls
                if state.hat_γ < iter.tol_b / iter.N
                    @warn "parameter `γ` became too small in ls 4 ($(state.hat_γ))"
                    return nothing
                end
                prox!(state.tz, iter.g, state.s, state.hat_γ) # \tilde z^k_i

                fi_u = gradient!(state.∇f_temp, iter.F[i], state.u) # grad eval
                global fi_tz = iter.F[i](state.tz)  

                ls_lhs = fi_tz - fi_u - real(dot(state.∇f_temp, state.tz .- state.u))
                ls_rhs = iter.N * norm(state.tz .- state.u)^2/(2*state.γ[i])

                # tol = 10^(-6)  * (1 + abs(ls_rhs)) # bug prone!
                # ls_lhs <= ls_rhs + tol && break  # the ls condition (Table 1, 8b)                
                ls_lhs <= ls_rhs + iter.ls_tol*(1+ls_rhs) && break  # the ls condition (Table 1, 8b)                
                println("ls 4")
                println("lhs : $(ls_lhs)  /  rhs : $(ls_rhs)")


                hat_γ_prev = state.hat_γ
                γ_prev = state.γ[i]
                state.γ[i] *= iter.η # update γ
                state.hat_γ = 1 / sum(1 ./ state.γ) # update hat_γ
                
                state.s ./= hat_γ_prev
                state.s .+= (1/state.γ[i] - 1/γ_prev) * state.u
                state.s .*= state.hat_γ
                
                reset!(state.H)
                state.z_prev .= zero(state.s)
                state.v_prev .= zero(state.s)
            end

            # iterates
            state.s .+= (state.hat_γ / iter.N) .* state.∇f_temp # updating s
            state.sum_∇fu .-= state.∇f_temp # update sum ∇f_i for next iter 
            gradient!(state.∇f_temp, iter.F[i], state.tz) 
            state.sum_∇fu .+= state.∇f_temp # update sum ∇f_i for next iter
            state.s .-= (state.hat_γ / iter.N) .* state.∇f_temp
            state.s .+= state.hat_γ * (state.tz .- state.u) ./ state.γ[i]

            # updates for the next linesearch (ls 1)
            state.sum_tz += state.tz / state.γ[i]
            state.sum_ftz += fi_tz
            state.sum_nrm_tz += norm(state.tz)^2 / state.γ[i]
            state.sum_innprod_tz += real(dot(state.∇f_temp, state.tz)) 

            state.A[i,:] .= state.tz
        end
    end
    println("norm tz: $(state.sum_nrm_tz)")
    println("min γ: $(minimum(state.γ)) - $(sum(state.A[1,:])) - $(sum(state.A[iter.N,:]))")

    # if norm(state.s) == 0
    #     println("hat_γ: $(state.hat_γ)")
    #     println("tz: $(norm(state.tz))")
    #     println("u: $(norm(state.u))")
    # end

    return state, state
end
solution(state::FINITO_lbfgs_adaptive_state) = state.z
epoch_count(state::FINITO_lbfgs_adaptive_state) = state.τ   # number of epochs is epoch_per_iter + log_β(τ) , where tau is from ls and epoch_per_iter is 3 or 4. Refer to it_counter function in utilities.jl
