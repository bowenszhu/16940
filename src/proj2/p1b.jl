using FastGaussQuadrature: gausslegendre
using LinearAlgebra: Symmetric, eigen!
using Statistics: mean, cov
using Plots: plot, plot!, savefig
using LaTeXStrings: @L_str
using Distributions: Normal
using ModelingToolkit, MethodOfLines, DifferentialEquations
const μy = 1.0
const L = 0.3
const σ²y = 0.3
const μf = -1.0
const σ²f = 0.2
const uᵣ = 1.0
c(x₁, x₂) = σ²y * exp(-abs(x₁ - x₂) / L)

function nystrom(n::Integer)
    x, w = gausslegendre(n)
    @. x = (x + 1.0) * 0.5
    @. w = sqrt(w / 2) # √w
    C = Symmetric(c.(x, x'))
    WCW = Symmetric(w .* C .* w')
    λ, ψ = eigen!(WCW; sortby = -)
    ψ ./= w
    x, w, λ, ψ, C
end

function total_mset_size(p::Integer, d::Integer)
    sum(binomial(i + d - 1, i) for i in 0:p; init = 0)
end
function total_degree_mset(p::Integer, d::Integer)
    sz = total_mset_size(p, d)
    mset = Matrix{Int}(undef, d, sz)
    mset_helper!(mset, p)
    mset
end
function mset_helper!(A, p)
    d = size(A, 1)
    if p == 0
        fill!(A, 0)
        return
    elseif d == 1
        A' .= 0:p
        return
    end
    col_st = 0
    col_end = 0
    for p_top in p:-1:0
        col_st = col_end + 1
        col_end += total_mset_size(p - p_top, d - 1)
        A[1, col_st:col_end] .= p_top
        mset_helper!(view(A, 2:d, col_st:col_end), p - p_top)
    end
end

uKL = Matrix{Float64}(undef, 11, 10)
for i in axes(uKL, 2)
    n = 10
    Z = randn(n)
    s, w, λ, ψ, C = nystrom(n)
    sλ = sqrt.(λ)
    psi(ss) = ((c.(ss, s) .* w)' * ψ)' ./ λ
    function yy(xx)
        ψx = psi(xx)
        @. ψx *= sλ * Z
        μy + sum(ψx)
    end
    dF = Normal(μf, sqrt(σ²f))
    F = rand(dF)
    @variables x u(..)
    D = Differential(x)
    k(x) = exp(yy(x))
    # @register_symbolic k(x)
    deq = [D(k(x) * D(u(x))) ~ -5.0]
    xₘᵢₙ = 0.0
    xₘₐₓ = 1.0
    domain = [x ∈ (xₘᵢₙ, xₘₐₓ)]
    bcs = [u(xₘₐₓ) ~ uᵣ, k(x) * D(u(xₘᵢₙ)) ~ -F]
    iv = [x]
    dv = [u(x)]
    N = 10
    dx = (xₘₐₓ - xₘᵢₙ) / N
    discretization = MOLFiniteDifference([x => dx])
    pdesys = PDESystem(deq, bcs, domain, iv, dv; name = :elliptic)
    prob = discretize(pdesys, discretization)
    sol = solve(prob)
    uKL[:, i] .= sol[u(x)]
end

plot(xlabel = L"x", ylabel = L"u(x,ω)", legend = false)
for i in axes(uKL, 2)
    plot!(0:0.1:1, @view uKL[:, i])
end
savefig("p1bKL.svg")
μu = vec(mean(uKL; dims = 2))
plot(0:0.1:1, μu, xlabel = L"x", ylabel = L"E[u(x,ω)]", legend = false)
savefig("p1bKLmean.svg")
@show CKL = cov(uKL[1:(end - 1), :]')
open("p1b.txt", "w") do io
    write(io, "covariance\n")
    write(io, "$CKL\n")
end
