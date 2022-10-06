using Distributions, StatsBase, Plots, LaTeXStrings

d = Pareto(3 / 2) # distribution
ns = 1 .<< (1:15)
v = similar(ns, Float64) # variance
s = similar(v) # skewness
k = similar(v) # kurtosis
trial = 1000
for (i, n) in enumerate(ns)
    x̄ₙ = [mean(rand(d, n)) for _ in 1:trial]
    m, v[i] = mean_and_var(x̄ₙ)
    s[i] = skewness(x̄ₙ, m)
    k[i] = kurtosis(x̄ₙ, m)
end
plot(ns, v, xaxis = :log, xlabel = L"n", ylabel = "variance", legend = false)
savefig("p1variance.png")
plot(ns, s, xaxis = :log, xlabel = L"n", ylabel = "skewness", legend = false)
savefig("p1skewness.png")
plot(ns, k, xaxis = :log, xlabel = L"n", ylabel = "kurtosis", legend = false)
savefig("p1kurtosis.png")
