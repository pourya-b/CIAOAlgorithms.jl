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

export Jac!


function Jac!(H::M, f::NormL1, x::AbstractArray{T}, gamma::Real) where {T, M <: AbstractArray{T}}
    y = similar(x)
    gl = gamma*f.lambda
    for i in eachindex(x)
        y[i] = ( ((x[i] <= -gl) || (x[i] >= gl)) ? 1 : 0)
    end
    H .= diagm(0 => y)
    nothing
end