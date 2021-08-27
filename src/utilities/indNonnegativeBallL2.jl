
using LinearAlgebra
include("approx_inequality.jl")
# export ProximableFunction
export prox, prox!

# abstract type ProximableFunction end
export IndNonnegativeBallL2

struct IndNonnegativeBallL2{R <: Real} 
    r::R
    function IndNonnegativeBallL2{R}(r::R) where {R <: Real}
        if r <= 0
            error("parameter r must be positive")
        else
            new(r)
        end
    end
end

is_convex(f::IndNonnegativeBallL2) = true
is_set(f::IndNonnegativeBallL2) = true

IndNonnegativeBallL2(r::R=1.0) where {R <: Real} = IndNonnegativeBallL2{R}(r)

function (f::IndNonnegativeBallL2)(x::AbstractArray{T}) where {R <: Real, T <: RealOrComplex{R}}
    if isapprox_le(norm(x), f.r, atol=eps(R), rtol=sqrt(eps(R)))
        for k in eachindex(x)
            if x[k] < 0
                return R(Inf)
            end
        end
        return R(0)
    end
    return R(Inf)
end

function prox!(y::AbstractArray{T}, f::IndNonnegativeBallL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    for k in eachindex(x)
        if x[k] < 0
            y[k] = R(0)
        else
            y[k] = x[k]
        end
    end
    
    scal = f.r/norm(y)
    if scal > 1
        return R(0)
    end
    for k in eachindex(y)
        y[k] = scal*y[k]
    end
    return R(0)
end

function prox(f::IndNonnegativeBallL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    y = copy(x)
    for k in eachindex(x)
        if x[k] < 0
            y[k] = R(0)
        else
            y[k] = x[k]
        end
    end
    
    scal = f.r/norm(y)
    if scal > 1
        return y, R(0)
    end
    for k in eachindex(y)
        y[k] = scal*y[k]
    end
    return y, R(0)
end

fun_name(f::IndNonnegativeBallL2) = "indicator of an L2 norm ball and nonnegative orthant"
fun_dom(f::IndNonnegativeBallL2) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndNonnegativeBallL2) = "x ↦ 0 if ||x|| ⩽ r & all(0 ⩽ x), +∞ otherwise"
fun_params(f::IndNonnegativeBallL2) = "r = $(f.r)"

function prox_naive(f::IndNonnegativeBallL2, x::AbstractArray{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    y = copy(x)
    for k in eachindex(x)
        if x[k] < 0
            y[k] = R(0)
        end
    end
    
    normx = norm(y)
    if normx > f.r
        y = (f.r/normx)*y
    end
    return y, R(0)
end
