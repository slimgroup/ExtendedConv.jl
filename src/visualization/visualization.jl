# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: December 2020
# Copyright: Georgia Institute of Technology, 2020

export filter_normalization, sol_traj

function filter_normalization(theta::Flux.Params)

    d = deepcopy(theta)

    for (p, q) in zip(theta, d)

        if ndims(p) == 4

            q .= randn(Float32, size(p))

            for i in size(p, 4)
                q[:, :, :, i] .= q[:, :, :, i] / (Flux.norm(q[:, :, :, i]) + 1f-8)
                q[:, :, :, i] .= q[:, :, :, i] * Flux.norm(p[:, :, :, i])
            end

        end
    end

    return d
end


function +(theta::Flux.Params, delta::Flux.Params)

    d = deepcopy(theta)

    for (p, q) in zip(d, delta)

        if ndims(p) > 1
            p .= p + q
        end

    end

    return d
end

function *(alpha::Float32, theta::Flux.Params)

    d = deepcopy(theta)

    for p in d

        if ndims(p) > 1
            p .= alpha*p
        end

    end

    return d
end


function sol_traj(delta, eta, ws, w0)

    delta_ = cat([delta[i] for i=1:2:length(delta)]..., dims=3)[:, :, :, 1]
    eta_ = cat([eta[i] for i=1:2:length(eta)]..., dims=3)[:, :, :, 1]

    A = cat(vec(delta_), vec(eta_), dims=2)
    proj(v) = (A'*A)^(-1) * (A'*vec(v - w0))


    proj_a = []
    proj_b = []

    for i=1:length(ws)
        pa, pb = proj(ws[i])
        push!(proj_a, pa)
        push!(proj_b, pb)
    end

    return proj_a, proj_b
end
