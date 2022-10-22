using DelimitedFiles: readdlm
using Distributions: Bernoulli, Binomial
using Random: rand!
using Statistics: mean
using Plots: histogram, savefig, plot, plot!, histogram!, vline!

data = readdlm("baseball.txt"; header = true)[1]
M = size(data, 1)
P = @view data[:, 2]

# (a)
N = 90
trial = Matrix{Int}(undef, N, M)
for i in 1:M
    rand!(Bernoulli(P[i]), @view trial[:, i])
end
pᴺ = vec(mean(trial; dims = 1))
p̄ᴺ = mean(pᴺ)
σ₀² = p̄ᴺ * (1 - p̄ᴺ) / N
p̂ᴶˢ = pᴺ .- p̄ᴺ
p̂ᴶˢ .*= 1 - (M - 3) * σ₀² / sum(abs2, p̂ᴶˢ)
p̂ᴶˢ .+= p̄ᴺ
tseeᴺ = sum(abs2, pᴺ .- P)
tseeᴶˢ = sum(abs2, p̂ᴶˢ .- P)
open("p1a.txt", "w") do io
    write(io, "tseeᴺ $tseeᴺ\n")
    write(io, "tseeᴶˢ $tseeᴶˢ\n")
end

# (b)
n_trial = 100
pᴺ = Matrix{Float64}(undef, n_trial, M)
for i in 1:M
    rand!(Binomial(N, P[i]), @view pᴺ[:, i])
end
pᴺ ./= N
p̄ᴺ = vec(mean(pᴺ, dims = 2))
σ₀² = @. p̄ᴺ * (1 - p̄ᴺ) / N
p̂ᴶˢ = pᴺ .- p̄ᴺ
denominator = vec(sum(abs2, p̂ᴶˢ; dims = 2))
@. p̂ᴶˢ *= 1 - (M - 3) * σ₀² / denominator
p̂ᴶˢ .+= p̄ᴺ
seeᴺ = vec(sum(abs2, pᴺ .- P'; dims = 2))
seeᴶˢ = vec(sum(abs2, p̂ᴶˢ .- P'; dims = 2))
histogram(seeᴺ, fillalpha = 0.2, label = "vanilla Monte Carlo",
          normalize = true, title = "squared estimation error")
histogram!(seeᴶˢ, fillalpha = 0.2, normalize = true,
           label = "James-Stein estimation")
savefig("p1b.svg")

# (b) (i)
p̄ᴺ_player = vec(mean(pᴺ; dims = 1))
mseᴺ = vec(mean(abs2, pᴺ .- p̄ᴺ_player'; dims = 1))
plot(1:M, mseᴺ, label = "vanilla Monte Carlo", xlabel = "player",
     ylabel = "mean squared error", xticks = 1:M)
p̄ᴶˢ_player = vec(mean(p̂ᴶˢ; dims = 1))
mseᴶˢ = vec(mean(abs2, p̂ᴶˢ .- p̄ᴶˢ_player'; dims = 1))
plot!(1:M, mseᴶˢ, label = "James-Stein estimation")
savefig("p1b1.svg")

# (b) (ii)
plot(1:M, P, xlabel = "player", label = "Pᵢ real")
plot!(1:M, p̄ᴺ_player, label = "pᴺ average")
plot!(1:M, p̄ᴶˢ_player, label = "p̂ᴶˢ average")
savefig("p1b2.svg")

# (b) (iii)
plts = []
for i in 1:M
    plt = histogram(view(pᴺ, :, i), label = "pᴺ", title = "P[$i] = $(P[i])",
                    normalize = true, fillalpha = 0.2, titlefont = 40,
                    legendfont = 40)
    histogram!(plt, view(p̂ᴶˢ, :, i), label = "p̂ᴶˢ", normalize = true,
               fillalpha = 0.2)
    vline!(plt, [P[i]], line = :red, linewidth = 7, label = "Pᵢ")
    push!(plts, plt)
end
plot(plts..., layout = (6, 3), size = (4000, 5000))
savefig("p1b3.svg")
