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


function Hessian!(H::M, f::LeastSquares, x::AbstractArray{T}) where {T,M <: AbstractMatrix{T}}
    H .= f.A' * f.A
    H .*= f.lambda
    return H * x 
end


