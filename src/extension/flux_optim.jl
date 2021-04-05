export loss_flux

function loss_flux(func_val, dw, w, z, y, G)


    for (i, p) in enumerate(Flux.params(G))
        # We only update conv weights, so we ignore biases
        if length(size(p)) == 4
            p .= deepcopy(w[:, :, Int((i+1)/2)])
        end
    end

    func_val, back = Flux.pullback(Flux.params(G)) do
        5f-1*norm(G(z) - y)^2f0
    end
    grads = back(1.0f0)

    # Again, only extracing conv weight gradients and ignoring bias gradients
    for (i, p) in enumerate(Flux.params(G))
        if length(size(p)) == 4
            dw[:, :, Int((i+1)/2)] .= grads[p][:, :, 1, 1]
        end
    end

    return func_val
end
