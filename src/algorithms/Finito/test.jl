struct test{R<:Real} # {} declares the condition on the type, it can be omitted, but if omitted, one can only constrain x and y to have the same types, not any other conditions on the type itself!
    x::R
    y::R
    test{R}(;x=4,y=7) where {R<:Real} = new(x,y)
end
#
obj = test{Int64}(x=3)
println("test ...")
println(obj.y)

test(::Type{R}; kwargs...) where {R} = test{R}(;kwargs...)
test(;kwargs...) = test(Int64;kwargs...)
test_(;kwargs...) = test{Int64}(;kwargs...)
# test_(;x::R , y::R) where{R<:Real} = test{R}(;x,y)
#
obj2 = test(Int64;x=2)
obj3 = test(x=3)
obj4 = test_(x=0)

function change_x!(y::Vector{Int64})
    y[1] = -1
end
x = [5,4]
change_x!(x)
println(x)

# function change_x!(x::Int64) #? how to make it mutable?
#     x = 1
# end
# x = 6
# change_x!(x)
# println(x)

# y = copy(x) #? is copy necessary?
# x[1] = 200
# println(y)
# println(x)

mutable struct x_m
    xx
end

function change_x!(x::x_m)
    x.xx = 1
end
x = x_m(10)
change_x!(x)
println(x.xx)
x.xx = 3
println(x.xx)
