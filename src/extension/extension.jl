export Conv_cds
export MVP_cds
export extended_obj
export extended_obj_grad!
export extended_conv_op
export var_proj_optim
export var_proj_op

mutable struct Conv_cds
    diagonals
    offsets
    mask
end

(C::Conv_cds)(x::Vector{TF}) where TF = σ.(MVP_cds(C, x))
(C::Conv_cds)(w::Array{TF, 2}) where TF = Conv_cds(w, C.offsets, C.mask)


function MVP_cds(
    R::Array{TF, 2}, offset::Vector{TI}, x::Vector{TF}
) where {TF<:Real, TI<:Integer}

    return sum(R[:, i] .* circshift(x, -offset[i]) for i=1:length(offset))
end

MVP_cds(C::Conv_cds, x::Vector{TF}) where TF = MVP_cds(C.diagonals, C.offsets, x)


function extended_obj(
    x::Vector{TF}, y::Vector{TF}, C::Conv_cds, w_ex::Array{TF, 2}, w::Array{TF,2}, λ
) where {TF<:Real}

    return 5f-1*norm(y - C(w_ex)(x))^2 + 5f-1*λ^2*norm(C.mask.*(w_ex .- vec(w)'))^2
end

function extended_obj(
    x::Vector{TF}, y::Vector{TF}, C::Chain, w_ex::Array{TF, 3}, w::Array{TF,3}, λ
) where {TF<:Real}

    nlayers = length(C)
    Cloc = Chain([C[i](w_ex[:, :, i]) for i=1:nlayers]...)

    obj = 5f-1*norm(y - Cloc(x))^2
    obj += 5f-1*λ^2*sum(norm(C[i].mask.*(w_ex[:, :, i].- vec(w[:, :, i])'))^2 for i=1:nlayers)
    return obj
end


function extended_obj_grad!(grad, x, y, C, w_ex, w, λ)
    grad .= C[1].mask .* gradient(w_ex->extended_obj(x, y, C, w_ex, w, λ), w_ex)[1]
end


function var_proj_optim(x, y, C, w_ex_0, w, λ, opt_var_proj)
    # Do outer loop
    ext_obj(w_ex) = extended_obj(x, y, C, w_ex, w, λ)
    ext_obj_grad!(grad, w_ex) = extended_obj_grad!(grad, x, y, C, w_ex, w, λ)
    opt_ex = optimize(ext_obj, ext_obj_grad!, w_ex_0, LBFGS(m=100), opt_var_proj)

    # Minimizer for extended weights w_ex
    w_ex_est= Optim.minimizer(opt_ex)

    func_val = ext_obj(w_ex_est)

    return func_val, w_ex_est
end


function var_proj_op(func_val, dw, w, x, y, C, w_ex_0, λ, opt_var_proj)
    # Do outer loop
    ext_obj(w_ex) = extended_obj(x, y, C, w_ex, w, λ)
    ext_obj_grad!(grad, w_ex) = extended_obj_grad!(grad, x, y, C, w_ex, w, λ)
    opt_ex = optimize(ext_obj, ext_obj_grad!, w_ex_0, LBFGS(m=100), opt_var_proj)

    # Minimizer for extended weights w_ex
    w_ex_est= Optim.minimizer(opt_ex)

    func_val = ext_obj(w_ex_est)

    dw .= gradient(w0->extended_obj(x, y, C, w_ex_est, w0, λ), w)[1]
    func_val = ext_obj(w_ex_est)
    func_val
end

function solve_reduced_obj(proj_sol, C, w)
    w_ = zero(w)
    for j = 1:size(w, 3)
        sol = sum(C[j].mask.*proj_sol[:, :, j], dims=1)./sum(C[j].mask, dims=1)
        sol = reshape(sol, size(w)[1:2])
        w_[:, :, j] .= deepcopy(sol)
    end
    return w_
end


function extended_conv_op(nx, ny; nc=1, batchsize=1, k=3, stride=1, pad=1, nlayers=4, w=nothing)

    W = ones(Float32, k, k, 1, 1)
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

    # @assert islinear(K)[1]
    # @assert islinear(transpose(K))[1]
    # @assert isadjoint(K)[1]

    K = K*jo_eye(Float32, nx*ny)
    K = sparse(K)

    R0 , offset = mat2CDS(K)
    I = findall(x->x==0, R0)

    M = ones(Float32, size(R0))
    M[I] .= 0f0

    if w == nothing
        C = Chain([Conv_cds(R0.*randn(Float32, 1, k*k), offset, M) for i=1:nlayers]...)
    else
        C = Chain([Conv_cds(R0.*reshape(w[:, :, i], 1, k*k), offset, M) for i=1:nlayers]...)
    end

    return C

end
