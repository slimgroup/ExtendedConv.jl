using DrWatson
@quickactivate :ExtendedConv

using PyPlot
using Seaborn
using Random
using Flux
using JOLI
using Flux, Flux.Data.MNIST

set_style("whitegrid")
rc("font", family="serif", size=10)
sfmt=matplotlib.ticker.ScalarFormatter(useMathText=true)
sfmt.set_powerlimits((0, 0))
plt.ioff()

sim_name = "schematics"
save_path = plotsdir(sim_name)

nx = 8
ny = 8
nc = 1
batchsize = 1

# Conv Op
w = collect(range(-0.9999f0, 1f0, length=9))
W = reshape(w, 3, 3, 1, 1)
bias = zeros(Float32, 1)

G = Conv(W, bias, identity; stride=1, pad=1)
Gadj = ConvTranspose(W, bias, identity; stride=1, pad=1)

N = nx * ny * nc * batchsize
K = joLinearFunctionFwd_T(
    N, N,
    x -> vec(G(reshape(x, nx, ny, nc, batchsize))),
    y -> vec(Gadj(reshape(y, nx, ny, nc, batchsize))),
    Float32,Float32, name="Conv"
)

# Form explicit convolution matrix
K=K*jo_eye(Float32,nx*ny)

K_vis = deepcopy(K)
K_vis[K_vis.==0f0] .= NaN

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    W[end:-1:1, end:-1:1, 1, 1], cmap="tab20",
    vmin=-1, vmax=1)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "3x3-kernel.png"), fig)
close(fig)

# Toeplitz matrix
fig, ax = subplots(1, 1, figsize=(4, 4))
ax.imshow(
    K_vis, cmap="tab20",
    vmin=-1, vmax=1)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "toeplitz.png"), fig)
close(fig)


# Matricize multiple images
Random.seed!(13)
x1 = 1f0*ones(Float32, nx, ny) + 7f-1*randn(Float32, nx, ny)
x2 = 2f0*ones(Float32, nx, ny) + 7f-1*randn(Float32, nx, ny)
x3 = 3f0*ones(Float32, nx, ny) + 7f-1*randn(Float32, nx, ny)

X = hcat(vec(x1), vec(x2), vec(x3))

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    x1, cmap="Set3",
    vmin=0, vmax=4)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "image-1.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    x2, cmap="Set3",
    vmin=0, vmax=4)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "image-2.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    x3, cmap="Set3",
    vmin=0, vmax=4)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "image-3.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    X, cmap="Set3", aspect=1,
    vmin=0, vmax=4)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "X.png"), fig)
close(fig)


# MNIST images
train_imgs = MNIST.images()
X = Array{Float32}(undef, size(train_imgs[1])..., 1, 3)
Y = Array{Float32}(undef, size(train_imgs[1])..., 1, 3)

x1 = Float32.(train_imgs[1])
x2 = Float32.(train_imgs[2])
x3 = Float32.(train_imgs[3])

X = hcat(vec(x1), vec(x2), vec(x3))

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    x1, interpolation="nearest", cmap="gray",
    vmin=0, vmax=1)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "mnist-1.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    x2, interpolation="nearest", cmap="gray",
    vmin=0, vmax=1)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "mnist-2.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    x3, interpolation="nearest", cmap="gray",
    vmin=0, vmax=1)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "mnist-3.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    X, aspect=.1, interpolation="nearest", cmap="gray",
    vmin=0, vmax=1)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "X-mnist.png"), fig)
close(fig)


# MNIST Conv
nx, ny = size(x1)
nc = 1
batchsize = 1

# Conv Op
Random.seed!(1)
w = randn(Float32, 3, 3)/sqrt(9f0)
W = reshape(w, 3, 3, 1, 1)
bias = zeros(Float32, 1)

G = Conv(W, bias, identity; stride=1, pad=1)
Gadj = ConvTranspose(W, bias, identity; stride=1, pad=1)

N = nx * ny * nc * batchsize
K = joLinearFunctionFwd_T(
    N, N,
    x -> vec(G(reshape(x, nx, ny, nc, batchsize))),
    y -> vec(Gadj(reshape(y, nx, ny, nc, batchsize))),
    Float32,Float32, name="Conv"
)

# Form explicit convolution matrix
K=K*jo_eye(Float32,nx*ny)
Y = K*X

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    reshape(Y[:, 1], nx, ny), interpolation="nearest", cmap="gray",
    vmin=0, vmax=.2)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "y-1.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    reshape(Y[:, 2], nx, ny), interpolation="nearest", cmap="gray",
    vmin=0, vmax=.2)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "y-2.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    reshape(Y[:, 3], nx, ny), interpolation="nearest", cmap="gray",
    vmin=0, vmax=.2)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "y-3.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize=(4, 4))
im = ax.imshow(
    Y, aspect=.1, interpolation="nearest", cmap="gray",
    vmin=0, vmax=.2)
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
wsave(joinpath(save_path, "Y.png"), fig)
close(fig)
