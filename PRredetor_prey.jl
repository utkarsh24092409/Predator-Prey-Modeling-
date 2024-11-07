# In the following code, only the interaction terms will be modelled using UDEs/NNs. It makes the problem/code more interesting!
# In the following code, the model was trained only for a single predator-prey cycle, i.e., t = (0.0, 3.5)
# The UDE model's extrapolation beyond this time span was also analysed
# Plots for the number of rabbits and wolves were made - both for the training domain and the whole domain

#### Packages ####

using LinearSolve, DifferentialEquations, ComponentArrays, Random, Lux, Flux, DiffEqFlux, Optimization, OptimizationOptimisers, OptimizationOptimJL, Plots, Measures
# For more options of Loss Functions, we use LossFunctions.jl
using LossFunctions

# Changing to current directory
cd(@__DIR__)

#### Defining the ODE ####

# Initial conditions for rabbits, r(t), and wolves, w(t)
u0 = [1., 1,]

# Time span
tspan = (0., 10.)
# number of points to save at
# datasize = 100 # Possible to make it "length = datasize"
tsteps = range(tspan[1], tspan[2], length = 100)

# Parameters - α, β, γ, δ
params = (1.5, 1., 3., 1.) # α, β, γ, δ

# Set of ODEs
function LV_ODE(dP, u, p, t)
   # rabbits r and wolves w
   (r, w) = u
   # constants in the ODEs
   (α, β, γ, δ) = p
   # ODEs as variables
   dP[1] = α*r - β*r*w
   dP[2] = - γ*w + δ*r*w
end

# Defining the ODE Problem
# Note that the ODE Problem is solved for the whole tspan
LV_ode_prob = ODEProblem(LV_ODE, u0, tspan, params)

# Solving the ODE Problem
# Note that the ODE Problem is solved for the whole tspan and saved at all tsteps
LV_ode_sol = Array(solve(LV_ode_prob, AutoVern7(Rodas5()), reltol = 1e-6, abstol = 1e-6, saveat = tsteps))

#### UDE Approach ####

# Random number generator for NN parameters
rng = Random.default_rng()
Random.seed!(rng, 0)

# Using NNs to replace interaction terms involving both r and w
# To replace β*r*w term
# The below architecture worked better the best
NN1 = Lux.Chain(Lux.Dense(2, 16, relu), Lux.Dense(16, 16, relu), Lux.Dense(16, 1))
ps1, st1 = Lux.setup(rng, NN1)

# To replace γ*r*w term
NN2 = Lux.Chain(Lux.Dense(2, 16, relu), Lux.Dense(16, 16, relu), Lux.Dense(16, 1))
ps2, st2 = Lux.setup(rng, NN2)

# Parameters that we know
# assuming we only know α and γ
# Defining the known constants in a Dictionary
# NOTE: remember to place a colon before the variable names, e.g., :α !!!
# const_params = Dict(:α => 1.5, :β => 3.0) # α, γ
# OR, directly
# const_params = (1.5, 3.0) # α, γ

# Creating a parameter vector for the UDE/NN optimization
# Note that the LV equation parameters, α and γ, are not optimised as they are assumed to be already known
# Only the NN parameters for the interaction terms involving r*w are being considered for optimization here
p0_vec = (layer_1 = ps1, layer_2 = ps2)

# Defining the UDE equations
function LV_UDE(dP, u, p, t)
   (r, w) = u
   # Destructuring the known parameters of the LV equations
   # global const_params
   # NOTE: The below lines are replaced by directly using the constants, α and γ, in the equations
   # (α, γ) = const_params # replacing this by directly inputting the constants in the ODE equations

   #Defining the NNs
   # Note: NN_ir([u0[2]], p_ir, st_ir) returns a tuple with the result and layer information
   # NN_ir([u0[2]], p_ir, st_ir)[1] returns a vector
   # NN_ir([u0[2]], p_ir, st_ir)[1][1] returns the scalar value contained in the vector !!!
   # abs(NN_ir([u0[2]], p_ir, st_ir)[1][1]) gives the absolute value
  
   NN1rw = abs(NN1([r,w], p.layer_1, st1)[1][1])
   NN2wr = abs(NN2([r,w], p.layer_2, st2)[1][1])
   # Defining the modified LV equations
   # Using NNs to replace interaction terms involving both r and w
   dP[1] = 1.5*r - NN1rw
   dP[2] = -3.0*w + NN2wr
end

# Initial parameters to be passed on to the UDE problem / optimization solver
init_p = ComponentArray(p0_vec)

# Creating tspan1 for a single predator-prey cycle
# Training the model only for a single predator-prey cycle
# Defining the step-size of tsteps1 = step size of tsteps -> for calculating loss b/w UDE and ODE on a 1-to-1 basis
tspan1 = (0., 3.5)
tsteps1 = range(tspan1[1], tspan1[2], step = 0.10101010101010101)
#length(tsteps1) = 35

# Initialising the UDE problem
# Parameters init_p added here
LV_ude_prob = ODEProblem{true}(LV_UDE, u0, tspan1, init_p)

# Defining a UDE prediction function
# The extra parameter, t, is to specify any custom tspan - to check predictions at future values as well!
function LV_pred(θ, t)
   # NOTE: DO NOT FORGET TO SPECIFY 'p' (here, p = θ) as parameters for NN in the Array(solve()) function !!!
   pred = Array(solve(LV_ude_prob, AutoVern7(Rodas5()), p=θ, sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat = tsteps1))
   # The below code works only when the user specifies t
   # It modifies the original LV_ude_prob tspan to any user-defined time span
   # This helps in directly mentioning any time span in the LV_pred function
   if isempty(t) == 0
       LV_ude_prob1 = remake(LV_ude_prob, tspan = t)
       tsteps2 = range(t[1], t[2], step = 0.10101010101010101)
       pred = Array(solve(LV_ude_prob1, AutoVern7(Rodas5()), p=θ, sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat = tsteps2))
   end
end

# Defining a simple L2 loss function
# LV_ode_sol[1, 1:k] refers to the 'k' terms that can be correlated with LV_pred() on a 1-to-1 basis (k refers to length of tsteps)
function LV_ude_loss(ps)
   # Not using t in LV_pred(θ, t) here
   pred = LV_pred(ps)
   # Loss is calculated for both r and w, i.e., r => LV_ode_sol[1,:], w => LV_ode_sol[2,:] (same for pred)
   # Length of tsteps1, k = 35
   loss = sum(abs2, LV_ode_sol[1,1:35] .- pred[1,1:35]) + sum(abs2, LV_ode_sol[2,1:35] .- pred[2,1:35])
   return loss
end

# Callback function to monitor loss
iter = 0
function callback(p,l)
   global iter
   iter += 1
   # Prints every 250 iterations
   if iter%250 == 0
       println(l)
   end
   return false
end

#### Optimization ####

# Automatic Differentiation scheme
adtype = Optimization.AutoZygote()

# Defining the optimization function
opt_fn = Optimization.OptimizationFunction((x,p)->LV_ude_loss(x), adtype)

# Defining the Optimization problem
opt_prob = Optimization.OptimizationProblem(opt_fn, init_p)

# Solving the optimization problem
# Note that the parameters are saved at tsteps1, i.e., for tspan1 = (0.0, 3.5)
opt_sol = Optimization.solve(opt_prob, OptimizationOptimisers.Adam(0.00001), maxiters = 20000, callback = callback, saveat = tsteps1)

#### Further Optimization ####

# Re-defining the Optimization problem
opt_prob2 = remake(opt_prob, u0 = opt_sol.u)
# Re-solving the optimization problem
opt_sol2 = Optimization.solve(opt_prob2, OptimizationOptimisers.Adam(0.00005), maxiters = 20000, callback = callback, saveat = tsteps1)
# Re-defining the Optimization problem
opt_prob3 = remake(opt_prob2, u0 = opt_sol2.u)
# Re-solving the optimization problem
opt_sol3 = Optimization.solve(opt_prob3, OptimizationOptimisers.Adam(0.0005), maxiters = 50000, callback = callback, saveat = tsteps1)

#### Plotting ####

# Plot for no. of rabbits, r
plt_R = plot(tsteps, LV_ode_sol[1, :], lw = 3, legend_position = :topright, label = "Rabbits_ODE", ylabel = L"$Solution$", xlabel = L"$Time$ $(units)$", titlefont="Computer Modern", tickfont="Computer Modern", guidefont="Computer Modern", colorbar_titlefont="Computer Modern", legendfont="Computer Modern", plot_titlefont="Computer Modern");
scatter!(plt_R, tsteps, LV_pred(opt_sol3.u, tspan)[1, :], label = "Rabbits_UDE", title = "\nNumber of rabbits", ylabel = L"$Solution$", xlabel = L"$Time$ $(units)$", titlefont="Computer Modern", tickfont="Computer Modern", guidefont="Computer Modern", colorbar_titlefont="Computer Modern", legendfont="Computer Modern", plot_titlefont="Computer Modern");
ylabel!(L"$Population$");
ylims!(0., 10.);
# Plot for no. of wolves, w
plt_W = plot(tsteps, LV_ode_sol[2, :], lw = 3, legend_position = :topright, label = "Wolves_ODE", ylabel = L"$Solution$", xlabel = L"$Time$ $(units)$", titlefont="Computer Modern", tickfont="Computer Modern", guidefont="Computer Modern", colorbar_titlefont="Computer Modern", legendfont="Computer Modern", plot_titlefont="Computer Modern");
scatter!(plt_W, tsteps, LV_pred(opt_sol3.u, tspan)[2, :], label = "Wolves_UDE", title = "\nNumber of wolves", ylabel = L"$Solution$", xlabel = L"$Time$ $(units)$", titlefont="Computer Modern", tickfont="Computer Modern", guidefont="Computer Modern", colorbar_titlefont="Computer Modern", legendfont="Computer Modern", plot_titlefont="Computer Modern");
ylabel!(L"$Population$");
ylims!(0., 10.);
# Creating a plot
plot(plt_R, plt_W, layout = (2,1), size = (1000, 800), plot_title = "Lotka-Volterra Equations - Actual vs. UDE Approach \n- Prediction for the whole domain, tspan = (0.0, 10.0) -", left_margin = 2mm, right_margin = 2mm, top_margin = 2mm, bottom_margin = 2mm)
savefig("UDE_LV_beyond_training_domain.pdf")
# savefig("UDE_LV_within_training_domain.pdf") # For tspan1 / tsteps1

#### Miscellaneous ####
# length(LV_pred(opt_sol3.u)[1,:])

