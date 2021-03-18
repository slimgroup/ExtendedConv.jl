# Extended convolutional layers

This repository provides an preliminary Julia implementation for extended convolutional layers.

To start running the examples, clone the repository:

```bash
$ git clone https://github.com/slimgroup/ExtendedConv.jl
```

## Installation

This repository is based on [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl). Before running examples, install `DrWatson.jl` by:

```julia
pkg> add DrWatson
```

The only other manual installation is to make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

The necessary dependencies will be installed upon running your first experiment.

## Visualization loss landscape

To visualize the loss landscape for conventional and extended CNNs run the following script:

```bash
$ julia scripts/2d-ext-obj-visualization.jl
```

To plot the optimization trajectory on the landscape produced above, run the following:

To perform joint or conditional (posterior) samples via the pretrained normalizing flow (obtained by running the script above), run:

```bash
$ julia scripts/2d-ext-obj-visualization.jl
```

Running these scripts in order is required.

## Questions

Please contact alisk@gatech.edu or mlouboutin3@gatech.edu for further questions.


## Author

Ali Siahkoohi and Mathias Louboutin
