struct SVRG_basic_DNN_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Union{Array{Tf},Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    μ::Maybe{Union{Array{R},R}}  # convexity moduli of the gradients
    γ::Maybe{R}             # stepsize 
    m::Maybe{Int}           # number of inner loop updates
    plus::Bool              # for SVRG++ variant 
end

mutable struct SVRG_basic_DNN_state{R<:Real,Tx}
    γ::R                    # stepsize 
    m::Int                  # number of inner loop updates
    av::Tx                  # the running average
    z::Tx
    z_full::Tx              # the outer loop argument
    w::Tx                   # the inner loop variable
    ind::Array{Int}         # running idx set 
    # some extra placeholders 
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx
end

function SVRG_basic_DNN_state(γ::R, m, av::Tx, z::Tx, z_full::Tx, w::Tx, ind) where {R,Tx}
    return SVRG_basic_DNN_state{R,Tx}(γ, m, av, z, z_full, w, ind, copy(av), copy(av))
end

function Base.iterate(iter::SVRG_basic_DNN_iterable{R}) where {R}
    N = iter.N
    ind = collect(1:N)
    m = iter.m === nothing ? m = 2 * N : m = iter.m
    # updating the stepsize 
    if iter.γ === nothing
        if iter.plus
            @warn "provide a stepsize γ"
            return nothing
        else
            if iter.L === nothing || iter.μ === nothing
                @warn "smoothness or convexity parameter absent"
                return nothing
            else
                L_M = maximum(iter.L)
                μ_M = maximum(iter.μ)
                γ = 1 / (10 * L_M)
                # condition Theorem 3.1
                rho = (1 + 4 * L_M * γ^2 * μ_M * (N + 1)) / (μ_M * γ * N * (1 - 4L_M * γ))
                if rho >= 1
                    @warn "convergence condition violated...provide a stepsize!"
                end
            end
        end
    else
        γ = iter.γ # provided γ
    end
    # initializing the vectors 
    av = zero(iter.x0)
    for i = 1:N
        ∇f = (Flux.gradient(() -> iter.F(i,iter.x0), params(iter.x0)))[iter.x0][:,1]
        # ∇f, ~ = gradient(iter.F[i], iter.x0)
        ∇f ./= N
        av .+= ∇f
    end
    z_full = copy(iter.x0)
    z = zero(av)
    w = copy(iter.x0)
    state = SVRG_basic_DNN_state(γ, m, av, z, z_full, w, ind)
    print()
    return state, state
end

function Base.iterate(iter::SVRG_basic_DNN_iterable{R}, state::SVRG_basic_DNN_state{R}) where {R}
    # The inner cycle
    for i in rand(state.ind, state.m)
        state.temp .= (Flux.gradient(() -> iter.F(i,state.z_full), params(state.z_full)))[state.z_full][:,1]
        # gradient!(state.temp, iter.F[i], state.z_full)
        state.∇f_temp .= (Flux.gradient(() -> iter.F(i, state.w), params(state.w)))[state.w][:,1]
        # gradient!(state.∇f_temp, iter.F[i], state.w)
        state.temp .-= state.∇f_temp
        state.temp .-= state.av
        state.temp .*= state.γ
        state.temp .+= state.w
        CIAOAlgorithms.prox!(state.w, iter.g, state.temp, state.γ)
        state.z .+= state.w   # keeping track of the sum of w's
    end
    # full update 	
    state.z_full .= state.w #state.z ./ state.m
    # iter.plus || (state.w .= state.z_full) # only for basic SVRG
    state.z = zero(state.z)  # for next iterate 
    state.av .= state.z
    for i = 1:iter.N
        state.∇f_temp .= (Flux.gradient(() -> iter.F(i, state.z_full), params(state.z_full)))[state.z_full][:,1]
        # gradient!(state.∇f_temp, iter.F[i], state.z_full)
        state.∇f_temp ./= iter.N
        state.av .+= state.∇f_temp
    end
    # iter.plus && (state.m *= 2) # only for SVRG++

    return state, state
end


solution(state::SVRG_basic_DNN_state) = state.z_full
