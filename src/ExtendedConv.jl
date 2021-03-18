# Author: Ali Siahkoohi, alisk@gatech.edu
#         Mathias Louboutin
# Date: December 2020
# Copyright: Georgia Institute of Technology, 2020

module ExtendedConv

using DrWatson
import Pkg; Pkg.instantiate()

# Utilities
include("./utils/savefig.jl")

# Filter-wise normalization method
include("./visualization/visualization.jl")

include("./extension/extension.jl")
include("./extension/flux_optim.jl")

end
