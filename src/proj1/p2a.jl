using Random, Statistics, Plots, LaTeXStrings

g(x) = exp(-x^2 / 2) / sqrt(2π)
πᵗ(x) = exp(-x^2 / 2) * (sin(6x)^2 + 3cos(x)^2 * sin(4x)^2 + 1)
C = 5sqrt(2π)

n_target = 100_000
n = 0
t = 0
while n < n_target
    y = randn()
    u = rand()
    uCgy = g(y) * C * u
    πy = πᵗ(y)
    if uCgy < πy
        global n += 1
    end
    global t += 1
end
println("n = ", n)
println("t = ", t)
io = open("p2a.txt", "w")
write(io, "n = $n\n")
write(io, "t = $t\n")

t = 100_000
trial = 10_000
n = Vector{Int}(undef, trial)
Iⁿ⁽ᵗ⁾ = similar(n, Float64)
y = Vector{Float64}(undef, t)
uCgy = similar(y)
u = similar(y)
πy = u # reuse memory
accept = similar(uCgy, Bool)
for i in 1:trial
    randn!(y) # y ~ g
    map!(πᵗ, πy, y) # π(y)
    map!(g, uCgy, y) # g(y)
    rand!(u) # u ~ U(0, 1)
    @. uCgy *= u * C # u * C * g(y)
    map!(<, accept, uCgy, πy) # u * C * g(y) < π(y)
    n[i] = sum(accept)
    Iⁿ⁽ᵗ⁾[i] = mean(abs2, y[accept])
end
var_Iⁿ⁽ᵗ⁾ = var(Iⁿ⁽ᵗ⁾)
println("var(Iⁿ⁽ᵗ⁾) = ", var_Iⁿ⁽ᵗ⁾)
write(io, "var(Iⁿ⁽ᵗ⁾) = $var_Iⁿ⁽ᵗ⁾\n")
close(io)

histogram(n, legend = false, xlabel = L"n", title = L"t = %$t")
savefig("p2a_n.svg")
histogram(Iⁿ⁽ᵗ⁾, legend = false, xlabel = L"\hat I^{n(t)}_\mathrm{AR}", title = L"t = %$t")
savefig("p2a_I.svg")
