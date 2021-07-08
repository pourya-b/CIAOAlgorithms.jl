"""
**Burg kernel**
    Burg(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i log(x_i)
```
"""


export Hessian!, Hessian, Jac!, Jac


function Hessian(f::ProximableFunction, x)
    H = similar(x, length(x), length(x))
    # Hessian!(H, f, x)
    Hx = Hessian!(H, f, x)
    return H, Hx
end


function Jac(f::ProximableFunction, x, gamma)
    H = similar(x, length(x), length(x))
    # Jac!(H, f, x, gamma)
    Hx = Jac!(H, f, x, gamma)
    return H, Hx
end



# what about having different h_i? 

# prox is not returning function value for now!  
