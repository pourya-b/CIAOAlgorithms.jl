"""
**Burg kernel**
    Burg(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i log(x_i)
```
"""
# using ProximalOperators: LeastSquares
using ProximalOperators

export Hessian!


function Hessian!(Hes::M, sumobj::Sum, x::AbstractArray{R}) where {R <: Real, M <: AbstractMatrix{R}}
    
    #     # Hessian of sum is sum of gradients
    # val = R(0)
    # # to keep track of this sum, i may not be able to
    # # avoid allocating an array
    # Hes .= R(0)
    # temp = similar(Hes)
    # for f in sumobj.fs
    #     val += Hessian!(temp, f, x)
    #     Hes .+= temp
    # end
    # return val
    
    # Hessian of sum is sum of gradients
    # to keep track of this sum, i may not be able to
    # avoid allocating an array
    temp = similar(Hes)
    for f in sumobj.fs
        Hessian!(temp, f, x)
        Hes .+= temp
    end
    nothing
end


function Hessian!(H::M, g::Precompose, x::AbstractArray{R}) where {R <: Real, M <: AbstractMatrix{R}}
    res = g.L*x .+ g.b
    n = length(res)
    Hres = similar(res,n,n)
    Hessian!(Hres, g.f, res)
    H .= adjoint(g.L) * Hres * g.L
    nothing
end


# hessian for squared normL2
function Hessian!(H::M, f::SqrNormL2{S}, x::AbstractArray{R}) where {S <: Real, R <: Real, M <: AbstractMatrix{R}}
    n = length(x)
    H .= diagm(0 => ones(n) .* f.lambda)
    # H .= Matrix{Float64}(I, n, n) .* f.lambda
    return nothing
end

# Hessian logistic loss 
function Hessian!(H::M, f::LogisticLoss, x::AbstractArray{R}) where {R <: Real, M <: AbstractMatrix{R}}
    n = length(x)
    expyx = exp(dot(f.y, x))
    H .= diagm(0 => ones(n))
    H .*= f.mu* expyx / (1+expyx)^2
    # H .= Matrix{Float64}(I, n, n) .* (f.mu* expyx / (1+expyx)^2) 
    # println(H)
    nothing
end