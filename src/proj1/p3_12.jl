using ModelingToolkit, MethodOfLines, Distributions, Random,
      DifferentialEquations, StatsBase, Plots, LaTeXStrings
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
function diffusioneqn(F_, Y₁_, Y₂_, Y₃_, Y₄_)
    ps = [F => F_, Y₁ => Y₁_, Y₂ => Y₂_, Y₃ => Y₃_, Y₄ => Y₄_]
    pdesys = PDESystem(deq, bcs, domain, iv, dv, ps; name = :elliptic)
    prob = discretize(pdesys, discretization)
    solve(prob)
end
trial = 100
FYs = Matrix{Float64}(undef, trial, 6)
rand!(dF, @view FYs[:, 1])
rand!(dY, @view FYs[:, 2:5])
FYs[:, end] .= 1.0
uₓ₆ = Vector{Float64}(undef, trial)
for i in 1:trial
    sol = diffusioneqn(FYs[i, 1:5]...)
    uₓ₆[i] = sol(0.6)[1]
end
m, se = mean_and_std(uₓ₆)
α = 1.0 - 0.95
z = quantile(Normal(), 1 - α / 2)
zse = z * se
interval = m - zse, m + zse
open("p32.txt", "w") do io
    write(io, "mean $m\n")
    write(io, "standard error $se\n")
    write(io, "95% confidence interval $interval\n")
end
histogram(uₓ₆, legend = false, xlabel = L"u(x=0.6)", title = L"n=%$trial")
savefig("p32ab.svg")

β = FYs \ uₓ₆ # linear regression for control variate
rand!(dF, @view FYs[:, 1])
rand!(dY, @view FYs[:, 2:5])
for i in 1:trial
    sol = diffusioneqn(FYs[i, 1:5]...)
    uₓ₆[i] = sol(0.6)[1]
end
Y_control = FYs * β
Ym = mean(Y_control)
c = -cov(uₓ₆, Y_control) / varm(Y_control, Ym)
Z = Y_control
@. Z = uₓ₆ + c * (Y_control - Ym)
varZ = var(Z)
varX = var(uₓ₆)
open("p32.txt", "w") do io
    write(io, "varX $varX\n")
    write(io, "varZ $varZ\n")
end
