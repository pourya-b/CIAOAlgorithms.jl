module CIAOAlgorithms #how we use this module? always paths are saved in a same format/ any other formats? how to switch to dev mode? in the dev mode, what happens to the other packages? how to add a package to the dev mode?

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T, Nothing}

# utilities
include("utilities/indexingUtilities.jl")
include("utilities/IndexIterator.jl")
include("utilities/lbfgs.jl")
include("utilities/indNonnegativeBallL2.jl")
# algorithms
include("algorithms/Finito/Finito.jl")
include("algorithms/ProShI/ProShI.jl")
include("algorithms/SVRG/SVRG.jl")
include("algorithms/SGD/SGD.jl")
include("algorithms/SARAH/SARAH.jl")
include("algorithms/SAGA_SAG/SAGA.jl")

end # module
