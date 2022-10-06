using Random, Statistics, Plots, LaTeXStrings

g(x) = exp(-x^2 / 2) / sqrt(2π)
πᵗ(x) = exp(-x^2 / 2) * (sin(6x)^2 + 3cos(x)^2 * sin(4x)^2 + 1)
w̃(x) = πᵗ(x) / g(x)
h = abs2
C = 5sqrt(2π)
trial = 10_000

function var_AR(t::Integer)
    Îⁿ⁽ᵗ⁾ₐᵣ = Vector{Float64}(undef, trial)
    y = Vector{Float64}(undef, t)
    uCgy = similar(y)
    u = similar(y)
    πy = u # reuse memory
    accept = similar(y, Bool)
    for i in 1:trial
        randn!(y) # y ~ g
        map!(πᵗ, πy, y) # π(y)
        map!(g, uCgy, y) # g(y)
        rand!(u) # u ~ U(0, 1)
        @. uCgy *= u * C # u * C * g(y)
        map!(<, accept, uCgy, πy) # u * C * g(y) < π(y)
        Îⁿ⁽ᵗ⁾ₐᵣ[i] = mean(h, y[accept])
    end
    var(Îⁿ⁽ᵗ⁾ₐᵣ) * trial
end

function var_IS(t::Integer)
    Îᵗᵢₛ = Vector{Float64}(undef, trial)
    x = Vector{Float64}(undef, t)
    w̃x = similar(x)
    hxw̃x = x # reuse memory
    for i in 1:trial
        randn!(x)
        map!(w̃, w̃x, x) # w̃(x)
        map!(h, hxw̃x, x) # h(x)
        hxw̃x .*= w̃x # h(x)w̃(x)
        Îᵗᵢₛ[i] = sum(hxw̃x) / sum(w̃x)
    end
    var(Îᵗᵢₛ) * trial
end

t = 1 .<< (8:15)
var_Îⁿ⁽ᵗ⁾ₐᵣ = map(var_AR, t)
var_Îᵗᵢₛ = map(var_IS, t)

plot(t, var_Îⁿ⁽ᵗ⁾ₐᵣ, xlabel = L"t", ylabel = L"\mathrm{Var}", axis = :log,
     label = L"\hat I^{n(t)}_\mathrm{AR}", markershape = :circle)
plot!(t, var_Îᵗᵢₛ, label = L"\hat I^t_\mathrm{IS}", markershape = :circle)
savefig("p2c.svg")

g̃ᵣₑⱼ(x) = C * g(x) - πᵗ(x)
w̃ᵣₑⱼ(x) = πᵗ(x) / g̃ᵣₑⱼ(x)

function combo(t::Integer)
    ₙÎⁿ⁽ᵗ⁾ₐᵣ = Vector{Float64}(undef, trial)
    ₜ₋ₙÎᵗ⁻ⁿᵣₑⱼ = similar(ₙÎⁿ⁽ᵗ⁾ₐᵣ)
    y = Vector{Float64}(undef, t)
    uCgy = similar(y)
    u = similar(y)
    πy = u # reuse memory
    accept = similar(y, Bool)
    for i in 1:trial
        randn!(y) # y ~ g
        map!(πᵗ, πy, y) # π(y)
        map!(g, uCgy, y) # g(y)
        rand!(u) # u ~ U(0, 1)
        @. uCgy *= u * C # u * C * g(y)
        map!(<, accept, uCgy, πy) # u * C * g(y) < π(y)
        n = sum(accept)
        ₙÎⁿ⁽ᵗ⁾ₐᵣ[i] = mean(h, y[accept]) * n
        x = y[.!accept] # x ~ C * g - πᵗ
        w̃x = map(w̃ᵣₑⱼ, x) # w̃(x)
        hxw̃x = map(h, x) # h(x)
        hxw̃x .*= w̃x # h(x)w̃(x)
        ₜ₋ₙÎᵗ⁻ⁿᵣₑⱼ[i] = (sum(hxw̃x) / sum(w̃x)) * (t - n)
    end
    (ₙÎⁿ⁽ᵗ⁾ₐᵣ + ₜ₋ₙÎᵗ⁻ⁿᵣₑⱼ) / t
end

function var_combo(t::Integer)
    Îᵗ_combo = combo(t)
    var(Îᵗ_combo)
end

var_Îᵗ_combo = map(var_combo, t)
plot!(t, var_Îᵗ_combo, label = L"\hat I^t_\mathrm{combo}",
      markershape = :circle)
savefig("p2d.svg")
