using ModelingToolkit, MethodOfLines, Distributions, DifferentialEquations,
      StatsBase, Plots, LaTeXStrings
const μʸ = -1.0
const σ²ʸ = 1.0
const σʸ = √σ²ʸ
const μᶠ = -2.0
const σ²ᶠ = 0.5
const σᶠ = √σ²ᶠ
const uᵣ = 1.0
dY = Normal(μʸ, σʸ)
dF = Normal(μᶠ, σᶠ)
@variables x u(..)
@parameters F Y₁ Y₂ Y₃ Y₄
D = Differential(x)
Y(x) = ifelse(x < 0.25, Y₁, ifelse(x < 0.5, Y₂, ifelse(x < 0.75, Y₃, Y₄)))
k(x) = exp(Y(x))
s(_) = 5.0
deq = [D(k(x) * D(u(x))) ~ -s(x)]
xₘᵢₙ = 0.0
xₘₐₓ = 1.0
domain = [x ∈ (xₘᵢₙ, xₘₐₓ)]
bcs = [u(xₘₐₓ) ~ uᵣ, k(x) * D(u(xₘᵢₙ)) ~ -F]
iv = [x]
dv = [u(x)]
N = 10 # number of discretization intervals
dx = (xₘₐₓ - xₘᵢₙ) / N
discretization = MOLFiniteDifference([x => dx])
function diffusioneqn()
    ps = [
        F => rand(dF),
        Y₁ => rand(dY),
        Y₂ => rand(dY),
        Y₃ => rand(dY),
        Y₄ => rand(dY),
    ]
    pdesys = PDESystem(deq, bcs, domain, iv, dv, ps; name = :elliptic)
    prob = discretize(pdesys, discretization)
    solve(prob)
end
trial = 100
uₓ₆ = Vector{Float64}(undef, trial)
for i in 1:trial
    sol = diffusioneqn()
    uₓ₆[i] = sol(0.6)[1]
end
m, se = mean_and_std(uₓ₆)
α = 1.0 - 0.95
z = quantile(Normal(), 1 - α / 2)
zse = z * se
interval = m - zse, m + zse
open("p32ab.txt", "w") do io
    write(io, "mean $m\n")
    write(io, "standard error $se\n")
    write(io, "95% confidence interval $interval\n")
end
histogram(uₓ₆, legend = false, xlabel = L"u(x=0.6)", title = L"n=%$trial")
savefig("p32ab.svg")
