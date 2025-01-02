### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ b082f890-a6a6-11ef-15aa-adfb0dc2d2d1
begin
	using Pkg
	Pkg.activate(".") 
end

# ╔═╡ 38f8486b-ecb9-4411-89b1-240e87d4bb6a
begin 
	using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers , MKL, Random
	import ModelingToolkit: Interval, infimum, supremum, LinearAlgebra
	import Plots: plot, plot!, heatmap, scatter
	import JLD: save
	seed = 0
    Random.seed!(seed)
end

# ╔═╡ 14eea340-bfbe-42b2-9611-6301fab9f7af
md"$\begin{align}
	&u_t + c(x)u_x = 0, \text{ em } [0,5] \times [0,6.4], \\
	&u(x,0) = f(x), \\~\\
	&c(x) = 1/5+\sin(x-1)^2, \\
	&f(x) = e^{-100(x-1)^2}
\end{align}$"

# ╔═╡ 936dd5b6-356e-4bfe-afaa-04957bbab84a
Pkg.status()

# ╔═╡ 4386f819-361f-46f1-8239-b0ccda228ef2
begin
	xmin = 0; xmax = 5; tmin = 0; tmax = 6.4
	c(x) = 1/5+sin(x-1)^2
	f(x) = exp(-100(x-1)^2)
	aux(x,t) = ((t-2.0491*x) <= 1.4279) * ((t-2.0194*x) >= -4.9414)
	u_exata(x,t) = exp(-100*(atan((1/sqrt(6))*tan(atan(sqrt(6)*tan(x-1))-(sqrt(6)/5)*t)))^2)
end

# ╔═╡ 36c5c8de-7cda-43c1-b8ad-8a047e41022d
begin
    Neurons = 10
	Layers = 10
	Epoch = 10000

	N_D = 2000
	N_I = [1000]

	iter_ap = 1000
    n_pontos_substituidos = 600 
end

# ╔═╡ 96b2ad5a-fd45-4d0b-acf8-7e1a52d7ab78
begin
	@parameters x t
	@variables u(..)
	Dt = Differential(t)
	Dx = Differential(x)
	
	PDE = Dt(u(x,t)) + c(x)*Dx(u(x,t)) ~ 0

	Domain = [x ∈ Interval(xmin, xmax), t ∈ Interval(tmin, tmax)]
	
	Boundary_Condition = [u(x, 0) ~ f(x)]
end

# ╔═╡ daddd11f-d346-4f24-a268-90c8996df6c8
begin
Neural_Network = Chain(
	Dense(2, Neurons, Lux.tanh; 
	init_weight = Lux.glorot_uniform, init_bias = Lux.glorot_uniform),
		
    [Dense(Neurons, Neurons, Lux.tanh; 
	init_weight = Lux.glorot_uniform, init_bias = Lux.glorot_uniform) 
		for i in 1:1:Layers],
		
    Dense(Neurons, 1; 
	init_weight = Lux.glorot_uniform, init_bias = Lux.glorot_uniform))

	Weight_NN = Lux.setup(MersenneTwister(seed), Neural_Network)[1]
	
	Strategy = Adaptive_Points(N_D, N_I)  
 
	Discretization = PhysicsInformedNN(Neural_Network, Strategy, 
		init_params = Weight_NN)
 
	@named PDE_System = PDESystem(PDE, Boundary_Condition, Domain, [x, t], [u(x, t)])
	Problem = symbolic_discretize(PDE_System, Discretization)
end

# ╔═╡ f798744c-1c07-4597-9755-f18f46a01928
begin
	pde_loss_functions = Problem.loss_functions.pde_loss_functions
	bc_loss_functions = Problem.loss_functions.bc_loss_functions
	loss_functions = [pde_loss_functions; bc_loss_functions]
	
	loss_function(θ, p) = sum(map(l -> l(θ), loss_functions))
end

# ╔═╡ 2971c20f-1f97-415e-8a96-1b095f40e5a6
function callback(p, l)
	if p.iter%iter_ap == 0
		println("Iteracao: ", p.iter)
		println("loss: ", l)
		println("pde_losses: ", map(l_ -> l_(p.u), pde_loss_functions))
		println("bcs_losses: ", map(l_ -> l_(p.u), bc_loss_functions))
		Strategy.sets = Novospontos(n_pontos_substituidos, Strategy, Problem)
	end
	return false
end

# ╔═╡ 952f4903-f97f-43c2-a985-9d83cb3ac56a
begin
	f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
	Prob = Optimization.OptimizationProblem(f_, Problem.flat_init_params)
	res = Optimization.solve(Prob, ADAM(10^-3); callback = callback, maxiters = Epoch)
	phi = Problem.phi
end

# ╔═╡ 9bc34820-c507-4e61-9980-41b0c28eb09a
begin
	end_points = Strategy.sets[1]
	plot(scatter(end_points[1,:], end_points[2,:],
		mode="markers", legend=false, title="Inicial"))
end

# ╔═╡ c4b7ae9f-c3d9-4f45-af9d-dd3f3926a2c0
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

# ╔═╡ 1773b5a4-058e-4a14-9cab-5d51323a9d79
let
    d = 0.01; x = xmin:d:xmax
	
	u_rede = [first(phi([i, 0], res.minimizer)) for i in x]
	u_real = f.(x)

    erro = u_rede - u_real

    plot(x, u_real, linewidth=4, label="exata", line = :dash);
	plot!(x, u_rede, linewidth=4, label="rede");
end

# ╔═╡ Cell order:
# ╟─14eea340-bfbe-42b2-9611-6301fab9f7af
# ╠═b082f890-a6a6-11ef-15aa-adfb0dc2d2d1
# ╠═936dd5b6-356e-4bfe-afaa-04957bbab84a
# ╠═38f8486b-ecb9-4411-89b1-240e87d4bb6a
# ╠═4386f819-361f-46f1-8239-b0ccda228ef2
# ╠═36c5c8de-7cda-43c1-b8ad-8a047e41022d
# ╠═96b2ad5a-fd45-4d0b-acf8-7e1a52d7ab78
# ╠═daddd11f-d346-4f24-a268-90c8996df6c8
# ╠═f798744c-1c07-4597-9755-f18f46a01928
# ╠═2971c20f-1f97-415e-8a96-1b095f40e5a6
# ╠═952f4903-f97f-43c2-a985-9d83cb3ac56a
# ╠═9bc34820-c507-4e61-9980-41b0c28eb09a
# ╠═c4b7ae9f-c3d9-4f45-af9d-dd3f3926a2c0
# ╠═1773b5a4-058e-4a14-9cab-5d51323a9d79
