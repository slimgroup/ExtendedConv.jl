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

sim_name = "sim_name"
fname = "batchsize=1_f_tol=1e-6_iterations=100_k=3_lambda=0.04_n_grid=21_nlayers=4_nx=16_ny=16_seed1=21_seed2=12_z_std=0.01_#2.jld2"

file_path = datadir(sim_name, fname)
file = jldopen(file_path, "r")

n_grid = file["n_grid"]
lambda = file["lambda"]
nx = file["nx"]
ny = file["ny"]
k = file["k"]
nlayers = file["nlayers"]
z_std = file["z_std"]
f_tol = file["f_tol"]
iterations = file["iterations"]
batchsize = file["batchsize"]
w_true = file["w"]
z = file["z"]
y = file["y"]
v1 = file["v1"]
v2 = file["v2"]
loss = file["loss"]
loss_ext = file["loss_ext"]
αs = file["αs"]
βs = file["βs"]
seed1 = file["seed1"]
seed2 = file["seed2"]

nc = 1

G = Chain([Conv((k, k), nc => nc, sigmoid, pad=(1, 1), stride=(1, 1))  for i=1:nlayers]...)
Flux.loadparams!(G, deepcopy(w_true))

w_true_ext = deepcopy(cat([w_true[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1])

Random.seed!(19)
G = Chain([Conv((k, k), nc => nc, sigmoid, pad=(1, 1), stride=(1, 1))  for i=1:nlayers]...)

α = 2f0
β = -2f0
w0 = deepcopy(w_true) + α*v1 + β*v2
Flux.loadparams!(G, deepcopy(w0))
w0_ext = cat([w0[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1]

C = extended_conv_op(nx, ny; nc=nc, batchsize=1, k=k, nlayers=nlayers, w=w0_ext)
w_ex_0 = cat(
    [randn(Float32, size(C[i].mask)).*C[i].mask for i=1:length(C)]...,
    dims=3
)

max_itr = 200
adam_lr = 0.05

G_ext(x) = reshape(C(reshape(x, :)), nx, ny, nc, 1)[end:-1:1, end:-1:1]
perm_y(y, idx) = reshape(y[end:-1:1, end:-1:1, :, idx], :)

opt_var_proj = Optim.Options(
    iterations=iterations,
    f_tol=1f-6,
    show_trace=false,
    store_trace=false,
    show_every=1
)

var_proj!(func_val, dw, w0_ext) = var_proj_op(
    func_val, dw, w0_ext, reshape(z, :), perm_y(y, 1), C, w_ex_0,
    lambda, opt_var_proj
)

ws_ext_lbfgs = []
fvals_ext_lbfgs = []
dummy_f = 0f0
dummy_w = zeros(Float32, size(w0_ext))
cbb = tr -> begin
            push!(ws_ext_lbfgs, tr[end].metadata["x"])
            fval = var_proj!(dummy_w, dummy_w,  tr[end].metadata["x"])
            push!(fvals_ext_lbfgs, deepcopy(fval))
            false
        end

opt_global = Optim.Options(
    iterations=max_itr,
    show_trace=true,
    store_trace=true,
    show_every=1,
    extended_trace=true,
    callback=cbb
)

opt_lbfgs = optimize(Optim.only_fg!(var_proj!), w0_ext, LBFGS(m=100), opt_global)


w0 = deepcopy(w_true) + α*v1 + β*v2
w0_ext = cat([w0[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1]
Flux.loadparams!(G, deepcopy(w0))

lr = adam_lr
opt = ADAM(lr)
ws_ext = []
var_proj!(func_val, dw, w0_ext) = var_proj_op(
    func_val, dw, w0_ext, reshape(z, :), perm_y(y, 1), C, w_ex_0,
    lambda, opt_var_proj
)

fvals_ext = []
func_val = 0f0
dw = zeros(Float32, size(w0_ext))
for j =1:max_itr

    loss_val = var_proj!(func_val, dw, w0_ext)
    push!(ws_ext, deepcopy(w0_ext))
    println(" Iteration: ", j, " | Objective = ", loss_val, "\r")

    push!(fvals_ext, loss_val)
    Flux.Optimise.update!(opt, w0_ext, dw)
end


w0 = deepcopy(w_true) + α*v1 + β*v2
w0_ext = cat([w0[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1]
Flux.loadparams!(G, deepcopy(w0))

lr = adam_lr
opt = ADAM(lr)
ws = []
fvals = []
for i in 1:max_itr

    loss_val, back = Flux.pullback(Flux.params(G)) do
        5f-1*norm(G(z) - y)^2f0
    end
    grads = back(1.0f0)
    println(" Iteration: ", i, " | Objective = ", loss_val, "\r")

    what = deepcopy(Flux.params(G))
    what_ = cat([what[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1]
    push!(ws, what_)

    push!(fvals, loss_val)
    for p in Flux.params(G)
        if length(size(p)) == 4
            Flux.Optimise.update!(opt, p, grads[p])
        end
    end

end


G = Chain([Conv((k, k), nc => nc, sigmoid, pad=(1, 1), stride=(1, 1))  for i=1:nlayers]...)
w0 = deepcopy(w_true) + α*v1 + β*v2
Flux.loadparams!(G, deepcopy(w0))
w0_ext = cat([w0[i] for i=1:2:2*nlayers]..., dims=3)[:, :, :, 1]

loss_flux!(func_val, dw, w0_ext) = loss_flux(func_val, dw, w0_ext, z, y, G)

ws_lbfgs = []
fvals_lbfgs = []
dummy_f = 0f0
dummy_w = zeros(Float32, size(w0_ext))
cbb = tr -> begin
            push!(ws_lbfgs, tr[end].metadata["x"])
            fval = loss_flux!(dummy_w, dummy_w,  tr[end].metadata["x"])
            push!(fvals_lbfgs, deepcopy(fval))
            false
        end

opt_global = Optim.Options(
    iterations=max_itr,
    show_trace=true,
    store_trace=true,
    show_every=1,
    extended_trace=true,
    callback=cbb
)

opt_lbfgs = optimize(Optim.only_fg!(loss_flux!), w0_ext, LBFGS(m=100), opt_global)


proj_ws_ext_lbfgs = sol_traj(v1, v2, ws_ext_lbfgs, w_true_ext)
proj_ws_ext = sol_traj(v1, v2, ws_ext, w_true_ext)
proj_ws = sol_traj(v1, v2, ws, w_true_ext)
proj_ws_lbfgs = sol_traj(v1, v2, ws_lbfgs, w_true_ext)


save_dict = @strdict loss loss_ext lambda n_grid f_tol iterations batchsize z_std nlayers nx ny k v1 v2 seed1 seed2 w_true z y αs βs max_itr adam_lr α β ws_ext_lbfgs ws_ext ws ws_lbfgs fvals_ext fvals_ext_lbfgs fvals fvals_lbfgs
@tagsave(
    datadir(sim_name, savename(save_dict, "jld2"; digits=12)),
    save_dict;
    safe=true
)

loss = (loss .- minimum(loss))./maximum(loss)
loss_ext = (loss_ext .- minimum(loss_ext))./maximum(loss_ext)

save_path = plotsdir(sim_name, savename(save_dict; digits=12))

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(4, 4))
implt=ax.contourf(αs, βs, loss_ext, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
    vmin=-1f-1, vmax=1f0)
ax.quiver(
    proj_ws_ext_lbfgs[1][1:end-1], proj_ws_ext_lbfgs[2][1:end-1],
    proj_ws_ext_lbfgs[1][2:end]-proj_ws_ext_lbfgs[1][1:end-1],
    proj_ws_ext_lbfgs[2][2:end]-proj_ws_ext_lbfgs[2][1:end-1],
    scale_units="xy", angles="xy", scale=1,
    color="y"
)
ax.set_title("Extended loss landscape")
fig.colorbar(implt, ax=ax, fraction=0.057, pad=0.01, format=sfmt)
ax.scatter(
    βs[argmin(loss_ext)[2]], αs[argmin(loss_ext)[1]], s=30.0,
    color="#b80000", marker="*"
)
ax.set_ylabel(L"$\beta$")
ax.set_xlabel(L"$\alpha$")
safesave(joinpath(save_path, "loss_ext_and_opt_traj_lbfgs.png"), fig)
close(fig)

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(4, 4))
implt=ax.contourf(αs, βs, loss_ext, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
    vmin=-1f-1, vmax=1f0)
ax.quiver(
    proj_ws_ext[1][1:end-1], proj_ws_ext[2][1:end-1],
    proj_ws_ext[1][2:end]-proj_ws_ext[1][1:end-1],
    proj_ws_ext[2][2:end]-proj_ws_ext[2][1:end-1],
    scale_units="xy", angles="xy", scale=1,
    color="y"
)
ax.set_title("Extended loss landscape")
fig.colorbar(implt, ax=ax, fraction=0.057, pad=0.01, format=sfmt)
ax.scatter(
    βs[argmin(loss_ext)[2]], αs[argmin(loss_ext)[1]], s=30.0,
    color="#b80000", marker="*"
)
ax.set_ylabel(L"$\beta$")
ax.set_xlabel(L"$\alpha$")
safesave(joinpath(save_path, "loss_ext_and_opt_traj.png"), fig)
close(fig)

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(4, 4))
implt=ax.contourf(αs, βs, loss, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
    vmin=-1f-1, vmax=1f0)
ax.quiver(
    proj_ws[1][1:end-1], proj_ws[2][1:end-1],
    proj_ws[1][2:end]-proj_ws[1][1:end-1],
    proj_ws[2][2:end]-proj_ws[2][1:end-1],
    scale_units="xy", angles="xy", scale=1,
    color="y"
)
ax.set_title("Loss landscape")
fig.colorbar(implt, ax=ax, fraction=0.057, pad=0.01, format=sfmt)
ax.scatter(
    βs[argmin(loss)[2]], αs[argmin(loss)[1]], s=30.0,
    color="#b80000", marker="*"
)
ax.set_ylabel(L"$\beta$")
ax.set_xlabel(L"$\alpha$")
safesave(joinpath(save_path, "loss_and_opt_traj.png"), fig)
close(fig)

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(4, 4))
implt=ax.contourf(αs, βs, loss, 25, cmap=Seaborn.ColorMap("mako"), linewidths=1.5,
    vmin=-1f-1, vmax=1f0)
ax.quiver(
    proj_ws_lbfgs[1][1:end-1], proj_ws_lbfgs[2][1:end-1],
    proj_ws_lbfgs[1][2:end]-proj_ws_lbfgs[1][1:end-1],
    proj_ws_lbfgs[2][2:end]-proj_ws_lbfgs[2][1:end-1],
    scale_units="xy", angles="xy", scale=1,
    color="y"
)
ax.set_title("Loss landscape")
fig.colorbar(implt, ax=ax, fraction=0.057, pad=0.01, format=sfmt)
ax.scatter(
    βs[argmin(loss)[2]], αs[argmin(loss)[1]], s=30.0,
    color="#b80000", marker="*"
)
ax.set_ylabel(L"$\beta$")
ax.set_xlabel(L"$\alpha$")
safesave(joinpath(save_path, "loss_and_opt_traj_lbfgs.png"), fig)
close(fig)

fvals_ext_norm = fvals_ext./fvals_ext[1]
fvals_ext_norm_lbfgs = fvals_ext_lbfgs./fvals_ext_lbfgs[1]
fvals_norm = fvals./fvals[1]
fvals_norm_lbfgs = fvals_lbfgs./fvals_lbfgs[1]

fig, ax = plt.subplots(1, 1, figsize=(4, 7.5))
ax.semilogy(fvals_norm, label="conventional (Adam)")
ax.semilogy(fvals_norm_lbfgs, label="conventional (L-BFGS)")
ax.semilogy(fvals_ext_norm, label="extended (Adam)")
ax.semilogy(range(0, length(ws_ext_lbfgs)-1, length=length(fvals_ext_norm_lbfgs)), fvals_ext_norm_lbfgs, label="extended (L-BFGS)")
ax.legend(loc="upper right")
ax.set_xlabel("Iterations")
ax.set_xlim([0, max_itr])
ax.set_ylabel("Normalized loss")
ax.set_title("Extended vs conventional convergence")
safesave(joinpath(save_path, "loss-comparison.png"), fig)
close(fig)

true_dist_ext = [norm(ws_ext[i] - w_true_ext) for i =1:length(ws_ext)]./norm(w_true_ext)
true_dist = [norm(ws[i] - w_true_ext) for i =1:length(ws)]./norm(w_true_ext)

true_dist_ext_lbfgs = [norm(ws_ext_lbfgs[i] - w_true_ext) for i =1:length(ws_ext_lbfgs)]
true_dist_ext_lbfgs = true_dist_ext_lbfgs./norm(w_true_ext)


true_dist_lbfgs = [norm(ws_lbfgs[i] - w_true_ext) for i =1:length(ws_lbfgs)]
true_dist_lbfgs = true_dist_lbfgs./norm(w_true_ext)

fig, ax = plt.subplots(1, 1, figsize=(4, 7.5))
ax.plot(true_dist, label="conventional (Adam)")
ax.plot(true_dist_lbfgs, label="conventional (L-BFGS)")
ax.plot(true_dist_ext, label="extended (Adam)")
ax.plot(true_dist_ext_lbfgs, label="extended (L-BFGS)")
ax.legend(loc="lower right")
ax.set_xlabel("Iterations")
ax.set_ylabel("Relative weight recovery error")
ax.set_xlim([0, max_itr])
ax.set_title("error w.r.t. ground-truth weight")
safesave(joinpath(save_path, "true-error-comparison.png"), fig)
close(fig)
