# Plot the loss function on a random 1D line
# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: December 2020

using DrWatson
@quickactivate :ExtendedConv

using Flux
using PyPlot
using Seaborn
using Printf
using LinearAlgebra
using Optim
using JLD2
using Random
using ProgressMeter
seed1 = 21
seed2 = 12

Random.seed!(seed1)

matplotlib.use("Agg")
set_style("whitegrid")
rc("font", family="serif", size=10)
font_prop = matplotlib.font_manager.FontProperties(
    family="serif",
    style="normal",
    size=10
)
sfmt=matplotlib.ticker.ScalarFormatter(useMathText=true)
sfmt.set_powerlimits((0, 0))

sim_name = "2D-visualization"

nx = 16
ny = 16
nc = 1
batchsize = 1
k = 3
nlayers = 4
lambda = .04f0
f_tol = 1f-6
iterations = 100

G = Chain([Conv((k, k), nc => nc, sigmoid, pad=(1, 1), stride=(1, 1))  for i=1:nlayers]...)

w = deepcopy(Flux.params(G))
w_true = cat([w[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1]

C = extended_conv_op(nx, ny; nc=nc, batchsize=1, k=k, nlayers=nlayers, w=w_true)

z_std = 1f-2
z = z_std*randn(Float32, nx, ny, nc, batchsize)
y = G(z)


G_ext(x) = reshape(C(reshape(x, :)), nx, ny, nc, 1)[end:-1:1, end:-1:1]
perm_y(y, idx) = reshape(y[end:-1:1, end:-1:1, :, idx], :)

αs = range(-4f0, 4f0, step=4f-1)
βs = range(-4f0, 4f0, step=4f-1)
n_grid = length(αs)

Random.seed!(seed2)

opt_var_proj = Optim.Options(
    iterations=iterations,
    f_tol=f_tol,
    show_trace=false,
    store_trace=false,
    show_every=1
)

num_figs = 4
p = Progress(num_figs*n_grid^2)

for fig_idx = 1:num_figs
    v1 = filter_normalization(w)
    v2 = filter_normalization(w)

    loss = zeros(Float32, length(αs), length(βs))
    loss_ext = zeros(Float32, length(αs), length(βs))

    for (i, α) in enumerate(αs)
        for (j, β) in enumerate(βs)
            Base.flush(Base.stdout)

            idx = rand(1:batchsize)

            ŵ = w + α*v1 + β*v2
            Flux.loadparams!(G, ŵ)

            ŵ0 = cat([ŵ[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1]
            C = extended_conv_op(nx, ny; nc=nc, batchsize=1, k=k, nlayers=nlayers, w=ŵ0)
            w_ex_0 = cat(
                [randn(Float32, size(C[i].mask)).*C[i].mask for i=1:length(C)]...,
                dims=3
            )

            loss_ext[i, j] = var_proj_optim(
                reshape(z[:, :, :, idx:idx], :), perm_y(y, idx), C, w_ex_0, ŵ0, lambda,
                opt_var_proj
            )[1]

            loss[i, j] = 5f-1*norm(G(z[:, :, :, idx:idx]) - y[:, :, :, idx:idx])^2

            ProgressMeter.next!(
                p;
                showvalues = [
                    ("Figure number", fig_idx),
                    (:α, α),
                    (:β, β),
                    ("Objectibe", loss[i, j]),
                    ("Extended objective", loss_ext[i, j])
                ]
            )

        end
    end

    save_dict = @strdict loss loss_ext lambda n_grid f_tol iterations batchsize z_std nlayers nx ny k v1 v2 seed1 seed2 w z y αs βs
    save_path = plotsdir(sim_name, savename(save_dict; digits=12))

    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits=12)),
        save_dict;
        safe=true
    )

    loss = (loss .- minimum(loss))./maximum(loss)
    loss_ext = (loss_ext .- minimum(loss_ext))./maximum(loss_ext)

    extent = [minimum(αs), maximum(αs), maximum(βs), minimum(βs)]

    fig = figure("conventional", figsize=(4, 4))
    contour(αs, βs, loss, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
        vmin=0.0, vmax=1f0)
    title("Loss landscape")
    colorbar(fraction=0.057, pad=0.01, format=sfmt)
    scatter(
        βs[argmin(loss)[2]], αs[argmin(loss)[1]], s=30.0,
        color="#b80000", marker="*"
    )
    xlabel(L"$\beta$")
    ylabel(L"$\alpha$")
    safesave(joinpath(save_path, "loss.png"), fig)
    close(fig)

    fig = figure("conventional", figsize=(4, 4))
    contour(αs, βs, loss_ext, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
        vmin=0.0, vmax=1f0)
    title("Extended loss landscape")
    colorbar(fraction=0.057, pad=0.01, format=sfmt)
    scatter(
        βs[argmin(loss_ext)[2]], αs[argmin(loss_ext)[1]], s=30.0,
        color="#b80000", marker="*"
    )
    xlabel(L"$\beta$")
    ylabel(L"$\alpha$")
    safesave(joinpath(save_path, "loss_ext.png"), fig)
    close(fig)

    fig = figure("conventional", figsize=(4, 4))
    contourf(αs, βs, loss, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
        vmin=-1f-1, vmax=1f0)
    title("Loss landscape")
    cb = colorbar(fraction=0.057, pad=0.01, format=sfmt)
    scatter(
        βs[argmin(loss)[2]], αs[argmin(loss)[1]], s=30.0,
        color="#b80000", marker="*"
    )
    xlabel(L"$\beta$")
    ylabel(L"$\alpha$")
    safesave(joinpath(save_path, "loss_contourf.png"), fig)
    close(fig)

    fig = figure("conventional", figsize=(4, 4))
    contourf(αs, βs, loss_ext, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
        vmin=-1f-1, vmax=1f0)
    title("Extended loss landscape")
    colorbar(fraction=0.057, pad=0.01, format=sfmt)
    scatter(
        βs[argmin(loss_ext)[2]], αs[argmin(loss_ext)[1]], s=30.0,
        color="#b80000", marker="*"
    )
    xlabel(L"$\beta$")
    ylabel(L"$\alpha$")
    safesave(joinpath(save_path, "loss_ext_contourf.png"), fig)
    close(fig)

    fig = figure("trace", figsize=(3*1.5, 7.5*1.5))
    plt.plot(αs, loss[:, argmin(loss)[2]], color="b", ls="--")
    plt.plot(αs, loss_ext[:, argmin(loss_ext)[2]], color="b")

    plt.plot(βs, loss[argmin(loss)[1], :], color="k", ls="--")
    plt.plot(βs, loss_ext[argmin(loss_ext)[1], :], color="k")

    plt.plot(αs, diag(loss), color="r", ls="--")
    plt.plot(αs, diag(loss_ext), color="r")
    safesave(joinpath(save_path, "trace.png"), fig)
    close(fig)

end
