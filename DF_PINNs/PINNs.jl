### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 07a27d93-420d-4332-a8c0-e8935c41c1e3
begin
	using Pkg
	Pkg.activate(".") 
end

# ╔═╡ 6f0eba90-c8a5-11ef-2f25-631ccfc97d0f
begin
	using JLD
	Rede_Ajustada = load("Rede_Ajustada.jld")
end

# ╔═╡ ddf07806-e1de-4aa7-bd79-98c0c522b191
begin 
	using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers , MKL, Random
	import ModelingToolkit: Interval, infimum, supremum, LinearAlgebra
	import Plots: plot, plot!, heatmap, scatter
	import JLD: save
	seed = 0
    Random.seed!(seed)
end

# ╔═╡ 1d5aa29a-2ebb-4112-b8e0-699d9a1f3544
md"$\begin{align}
	&u_t + c(x)u_x = 0, \text{ em } [0,5] \times [0,6.4], \\
	&u(x,0) = f(x), \\~\\
	&c(x) = 1/5+\sin(x-1)^2, \\
	&f(x) = e^{-100(x-1)^2}
\end{align}$"

# ╔═╡ ddfe8868-3a82-4151-be76-016eb60df1dd
begin
	xmin = 0; xmax = 5; tmin = 0; tmax = 6.4
	c(x) = 1/5+sin(x-1)^2
	f(x) = exp(-100(x-1)^2)
	aux(x,t) = ((t-2.0491*x) <= 1.4279) * ((t-2.0194*x) >= -4.9414)
	u_exata(x,t) = exp(-100*(atan((1/sqrt(6))*tan(atan(sqrt(6)*tan(x-1))-(sqrt(6)/5)*t)))^2)
end

# ╔═╡ ccbd2660-dee6-4b9d-8f5e-6ba965b0dbdc
begin
	Neurons = Rede_Ajustada["Neurons"]
	Layers = Rede_Ajustada["Layers"]
	Epoch = 8500

	N_D = 2000
	N_I = [1000]
end

# ╔═╡ 7b0f3195-956e-4af2-a367-d65390ee16cc
begin
	@parameters x t
	@variables u(..)
	Dt = Differential(t)
	Dx = Differential(x)
	
	PDE = Dt(u(x,t)) + c(x)*Dx(u(x,t)) ~ 0

	Domain = [x ∈ Interval(xmin, xmax), t ∈ Interval(tmin, tmax)]
	
	Boundary_Condition = [u(x, 0) ~ f(x)]
end

# ╔═╡ 89a805b1-a944-421c-9b36-411efc1474fd
begin
Neural_Network = Chain(
	Dense(2, Neurons, Lux.tanh),
		
    [Dense(Neurons, Neurons, Lux.tanh)],
		
    Dense(Neurons, 1, Lux.tanh))
	
	Strategy = FixedStochasticTraining(N_D, N_I)  
 
	Discretization = PhysicsInformedNN(Neural_Network, Strategy, 
		init_params = Rede_Ajustada["Weight_NN"])
 
	@named PDE_System = PDESystem(PDE, Boundary_Condition, Domain, [x, t], [u(x, t)])
	Problem = symbolic_discretize(PDE_System, Discretization)
end

# ╔═╡ 600f1616-7cbf-43c7-9fc8-fc2e2a03fe44
begin
	pde_loss_functions = Problem.loss_functions.pde_loss_functions
	bc_loss_functions = Problem.loss_functions.bc_loss_functions
	loss_functions = [pde_loss_functions; bc_loss_functions]
	
	loss_function(θ, p) = sum(map(l -> l(θ), loss_functions))
end

# ╔═╡ a6de5aed-75f6-4c01-8213-7412a02e6fec
function callback(p, l)
	if p.iter%500 == 0
		println("Iteracao: ", p.iter)
		println("loss: ", l)
		println("pde_losses: ", map(l_ -> l_(p.u), pde_loss_functions))
		println("bcs_losses: ", map(l_ -> l_(p.u), bc_loss_functions))
	end
	return false
end

# ╔═╡ 90626421-b08a-4c0f-b7b9-bb47f81218c6
begin
	f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
	Prob = Optimization.OptimizationProblem(f_, Problem.flat_init_params)
	res = Optimization.solve(Prob, ADAM(10^-3); callback = callback, maxiters = Epoch)
	phi = Problem.phi
end

# ╔═╡ 5eb582d8-391a-430b-aa54-e0acb7061fb3
let
	d = 0.005; x = xmin:d:xmax; t = tmin:d:tmax

	u_real = [u_exata(i, j)*aux(i, j) for j in t, i in x]
	u_rede = [first(phi([i, j], res.minimizer)) for j in t, i in x]
	u_erro = abs.(u_rede .- u_real)

    vmin = min(minimum(u_real), minimum(u_rede))
	vmax = max(maximum(u_real), maximum(u_rede))

	h1 = heatmap(x, t, u_real, title="Analitica", clim=(vmin, vmax))
	h2 = heatmap(x, t, u_rede, title="PINN", clim=(vmin, vmax))
	h3 = heatmap(x, t, u_erro, title = "Erro")
	heatmap(h1, h2, h3)
end

# ╔═╡ 2df0bfe8-0979-4a20-9f68-388dd1c68273
let
    d = 0.01; x = xmin:d:xmax
	
	u_rede = [first(phi([i, 0], res.minimizer)) for i in x]
	u_real = f.(x)

    erro = u_rede - u_real

    plot(x, u_real, linewidth=4, label="exata", line = :dash);
	plot!(x, u_rede, linewidth=4, label="rede");
end

# ╔═╡ Cell order:
# ╟─1d5aa29a-2ebb-4112-b8e0-699d9a1f3544
# ╠═6f0eba90-c8a5-11ef-2f25-631ccfc97d0f
# ╠═07a27d93-420d-4332-a8c0-e8935c41c1e3
# ╠═ddf07806-e1de-4aa7-bd79-98c0c522b191
# ╠═ddfe8868-3a82-4151-be76-016eb60df1dd
# ╠═ccbd2660-dee6-4b9d-8f5e-6ba965b0dbdc
# ╠═7b0f3195-956e-4af2-a367-d65390ee16cc
# ╠═89a805b1-a944-421c-9b36-411efc1474fd
# ╠═600f1616-7cbf-43c7-9fc8-fc2e2a03fe44
# ╠═a6de5aed-75f6-4c01-8213-7412a02e6fec
# ╠═90626421-b08a-4c0f-b7b9-bb47f81218c6
# ╠═5eb582d8-391a-430b-aa54-e0acb7061fb3
# ╠═2df0bfe8-0979-4a20-9f68-388dd1c68273
