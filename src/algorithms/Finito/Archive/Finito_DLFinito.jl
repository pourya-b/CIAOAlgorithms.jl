struct FINITO_DLFinito_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg} <: CIAO_iterable
    F::Array{Tf}            # smooth term
    g::Tg                   # nonsmooth term
    x0::Tx                  # initial point
    N::Int                    # of data points in the finite sum problem
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i
    γ::Maybe{Union{Array{R},R}}  # stepsizes
    sweeping::Int8          # to only use one stepsize γ
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    inseq::Int              # extra memory needed = length of inner seq (excluding z)                        # exm: 1231231234234234... as memory = 3
    rep::Int                # number of reps for each index
end

mutable struct FINITO_DLFinito_state{R<:Real,Tx}
    γ::Array{R}             # stepsize parameter
    hat_γ::R                # average γ
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches
    z_M::Vector{Tx}
    # some extra placeholders
    ∇f_temp::Tx             # placeholder for gradients
    z_full::Tx
    inds::Array{Int}        # needed for shuffled only!
end

function FINITO_DLFinito_state(γ::Array{R}, hat_γ::R, av::Tx, ind, d, z_M) where {R,Tx} # why this function is needed?
    return FINITO_DLFinito_state{R,Tx}(
        γ,
        hat_γ,
        av,
        ind,
        d,
        z_M,
        zero(av),
        zero(av),
        collect(1:d),
    )
end

function Base.iterate(iter::FINITO_DLFinito_iterable{R}) where {R} # rewriting Base?
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
    # println(iter.inseq)
    z_M = [copy(av) for i = 1:(iter.inseq+1)]

    state = FINITO_DLFinito_state(γ, hat_γ, av, ind, cld(N, r), z_M)

    return state, state
end

function Base.iterate(
    iter::FINITO_DLFinito_iterable{R},
    state::FINITO_DLFinito_state{R},
) where {R}
    # full update
    prox!(state.z_full, iter.g, state.av, state.hat_γ)
    state.av .= (iter.N / state.hat_γ ) .* state.z_full
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.F[i], state.z_full) # update the gradient
        state.av .-= state.∇f_temp
    end
    state.av .*= state.hat_γ / iter.N
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled

    # iter.sweeping == 3 && @error "not supported" # shuffled
    # println(state.inds)
    # repeat = 2 # number of repeats of each seq
    # println(iter.rep)
    Ind = Index_iterable(1, iter.inseq, iter.rep, iter.N)

    # Ind = take(Ind, repeat * iter.N)

    for (it, ind) in Iterators.enumerate(Ind)
        # println(it)
        if mod(it, iter.inseq * iter.rep) == 1
            # println("------new cycle starts-----")
            for ell in 1:(iter.inseq+1)       # for initializing the inner loop
                copyto!(state.z_M[ell], state.z_full)
            end
        end
        # println(ind)
        prox!(state.z_M[ind.m_n], iter.g, state.av, state.hat_γ)

        gradient!(state.∇f_temp, iter.F[state.inds[ind.i]], state.z_M[ind.m_o]) # update the gradient
        state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
        gradient!(state.∇f_temp, iter.F[state.inds[ind.i]], state.z_M[ind.m_n]) # update the gradient
        state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
        state.av .+= (state.hat_γ / state.γ[state.inds[ind.i]]) .* (state.z_M[ind.m_n] .- state.z_M[ind.m_o])
    end

    return state, state
end

# not exactly the last z since we should look at z_M[ind.m_n]
solution(state::FINITO_DLFinito_state) = state.z_M[end]



# taking prox to be able to have consistent output without knowing the last z_M cell updated
# function solution(state::FINITO_DLFinito_state)
#     sol = copy(state.z_full)
#     prox!(sol, iter.g, state.av, state.hat_γ)
#     return sol
# end


# count(state::FINITO_DLFinito_state) = []
