using DelimitedFiles: readdlm
using Distributions: Bernoulli, Binomial
using Random: rand!
using Statistics: mean
using Plots: histogram, savefig

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
plt_seeᴺ = histogram(seeᴺ, legend = false,
                     title = "squared estimation error\nvanilla Monte Carlo")
plt_seeᴶˢ = histogram(seeᴶˢ, legend = false,
                      title = "squared estimation error\nJames-Stein estimation")
savefig(plt_seeᴺ, "see_N.svg")
savefig(plt_seeᴶˢ, "see_JS.svg")
