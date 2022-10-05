using Plots, LaTeXStrings

function exp_inverse(u, λ)
    -log(1 - u) / λ
end

λ = 1 // 2
trial = 100000
X₁s = Vector{Float64}(undef, trial)
X₂s = similar(X₁s)
for i in 1:trial
    u = rand()
    r² = exp_inverse(u, λ)
    r = √r²
    θ = rand() * 2
    sinθ, cosθ = sincospi(θ)
    X₁s[i] = r * cosθ
    X₂s[i] = r * sinθ
end

plt₁ = histogram(X₁s, normalize = true, legend = false, title = L"X_1")
savefig(plt₁, "X1.png")
plt₂ = histogram(X₂s, normalize = true, legend = false, title = L"X_2")
savefig(plt₂, "X2.png")

plt = scatter(X₁s, X₂s, aspect_ratio = 1, xlabel = L"X_1", ylabel = L"X_2", legend = false,
              markersize = 1)
savefig(plt, "X1X2.png")
