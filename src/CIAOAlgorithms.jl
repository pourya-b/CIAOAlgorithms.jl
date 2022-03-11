module CIAOAlgorithms

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T, Nothing}

# utilities
include("utilities/indexingUtilities.jl")
include("utilities/IndexIterator.jl")
include("utilities/lbfgs.jl")
# include("utilities/indNonnegativeBallL2.jl")

# algorithms
include("algorithms/Finito/Finito.jl")
include("algorithms/SVRG/SVRG.jl")
include("algorithms/SGD/SGD.jl")
include("algorithms/SARAH/SARAH.jl")
include("algorithms/SAGA_SAG/SAGA.jl")

end # module
