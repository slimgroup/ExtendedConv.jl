# Author: Ali Siahkoohi, alisk@gatech.edu
#         Mathias Louboutin
# Date: December 2020
# Copyright: Georgia Institute of Technology, 2020

module ExtendedConv

using Flux
using LinearAlgebra
using PyPlot: Figure
using JOLI
using SetIntersectionProjection
using SparseArrays
using Optim

import Base.+, Base.*
import DrWatson: _wsave

# Utilities
include("./utils/savefig.jl")

# Filter-wise normalization method
include("./visualization/visualization.jl")

include("./extension/extension.jl")
include("./extension/flux_optim.jl")

end
