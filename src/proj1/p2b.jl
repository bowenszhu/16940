using Random, Statistics, Plots, LaTeXStrings

g(x) = exp(-x^2 / 2) / sqrt(2π)
πᵗ(x) = exp(-x^2 / 2) * (sin(6x)^2 + 3cos(x)^2 * sin(4x)^2 + 1)
w̃(x) = πᵗ(x) / g(x)
h = abs2

t = 100_000
trial = 10_000
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
var_Îᵗᵢₛ = var(Îᵗᵢₛ) * trial
println("var(Îᵗᵢₛ) = ", var_Îᵗᵢₛ)

open("p2b.txt", "w") do io
    write(io, "var(Îᵗᵢₛ) = $var_Îᵗᵢₛ\n")
end

histogram(Îᵗᵢₛ, legend = false, xlabel = L"\hat I^t_\mathrm{IS}",
          title = L"t = %$t")
savefig("p2b_I.svg")
