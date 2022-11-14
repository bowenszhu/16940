using FastGaussQuadrature: gausslegendre
using LinearAlgebra: Symmetric, eigen!
using Statistics: mean
using Plots: plot, plot!, savefig
using LaTeXStrings: @L_str

const μy = 1.0
const L = 0.3
const σ²y = 0.3
const μf = -1.0
const σ²f = 0.2
const uᵣ = 1.0
s(_) = 5.0
c(x₁, x₂) = σ²y * exp(-abs(x₁ - x₂) / L)

function nystrom(n::Integer)
    x, w = gausslegendre(n)
    @. x = (x + 1.0) * 0.5
    @. w = sqrt(w / 2) # √w
    C = Symmetric(c.(x, x'))
    WCW = Symmetric(w .* C .* w')
    λ, ψ = eigen!(WCW; sortby = -)
    ψ ./= w
    x, λ, ψ, C
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

plot(xlabel = L"p", ylabel = "chaos coeff", yaxis = :log10,
     title = "PCE convergence")
for n in 4:10
    x, λ, ψ, C = nystrom(n)
    λψ = ψ .* sqrt.(λ)' # √(λᵢ)ψᵢ
    temp = vec(sum(λψ; dims = 2))
    @. temp = exp(μy + temp / 2) # exp(μy + ∑λψ²(x)/2)
    function coeff(α::AbstractVector)
        num = λψ .^ α'
        res = map(j -> prod(view(num, :, j)), axes(num, 2))
        dem = prod(factorial, α)
        @. res = temp * res / dem
        res
    end
    α = Vector{Int}(undef, n)
    p = 10
    Cₐ = Vector{Float64}(undef, p + 1)
    for i in 0:p
        fill!(α, i)
        Cₐ[i + 1] = mean(coeff(α))
    end
    @. Cₐ = abs(Cₐ)
    plot!(0:p, Cₐ, label = L"n=%$n")
end
plot!()
savefig("p1a.svg")
