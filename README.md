[![DOI](https://zenodo.org/badge/348947266.svg)](https://zenodo.org/badge/latestdoi/348947266)

# Extended convolutional layers

This repository provides an preliminary Julia implementation for extended convolutional layers. For a short technical report on extended convolutional layers click [here](https://github.com/slimgroup/ExtendedConv.jl/blob/main/doc/report.pdf).

To start running the examples, clone the repository:

```bash
$ git clone https://github.com/slimgroup/ExtendedConv.jl
$ cd ExtendedConv.jl/
```

## Installation

Before starting installing the required packages in Julia, make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

Next, run the following commands in the command line to install the necessary libraries and setup the Julia project:

```bash
julia -e 'using Pkg; Pkg.add("DrWatson")'
julia -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url = "https://github.com/slimgroup/SLIMregistryJL.git"))'
julia --project -e 'using Pkg; Pkg.instantiate()'
```

After the last line, the necessary dependencies will be installed.

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
