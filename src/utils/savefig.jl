# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: December 2020
# Copyright: Georgia Institute of Technology, 2020

export _wsave

using PyPlot: Figure
import DrWatson: _wsave

_wsave(s, fig::Figure; dpi::Int=300) = fig.savefig(s, bbox_inches="tight", dpi=dpi)
