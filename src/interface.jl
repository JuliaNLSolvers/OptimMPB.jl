## Glue code for Optim.jl and csminwel.jl
immutable OptimSolver <: MathProgBase.SolverInterface.AbstractMathProgSolver
    optimizer
    options
end

function OptimSolver{T<:Optimizer}(::Type{T}; kwargs...)
  OptimSolver(T(), kwargs)
end

MathProgBase.NonlinearModel(s::OptimSolver) = OptimMathProgModel(s; s.options...)

type OptimProblem
  out
  eval_f::Function
  eval_grad_f::Function
  sense::Symbol
end

type OptimMathProgModel <: MathProgBase.SolverInterface.AbstractNonlinearModel
    options
    inner::OptimProblem
    numVar::Int
    varLB::Vector{Float64}
    varUB::Vector{Float64}
    initial_x
    sense::Int32
    d::MathProgBase.SolverInterface.AbstractNLPEvaluator
    s::Optimizer
    solve_time::Float64
    function OptimMathProgModel(s; options...)
           model = new()
           model.options = options
           model.s = s.optimizer
           model
    end
end

MathProgBase.SolverInterface.setwarmstart!(m::OptimMathProgModel, x) = copy!(m.initial_x, x)

MathProgBase.SolverInterface.loadproblem!(m::OptimMathProgModel, numVar::Int, x_l, x_u,  sense::Symbol,
                      d::MathProgBase.SolverInterface.AbstractNLPEvaluator) =
                      MathProgBase.SolverInterface.loadproblem!(m, numVar, 0, x_l, x_u, [], [], sense, d)

MathProgBase.SolverInterface.loadproblem!(m::OptimMathProgModel, numVar::Int,  sense::Symbol,
                      d::MathProgBase.SolverInterface.AbstractNLPEvaluator) =
                      MathProgBase.SolverInterface.loadproblem!(m, numVar, 0, [-Inf for j in 1:numVar], [+Inf for j in numVar], [], [], sense, d)

function MathProgBase.SolverInterface.loadproblem!(m::OptimMathProgModel, numVar::Integer, numConstr::Integer,
                      x_l, x_u, g_lb, g_ub, sense::Symbol, d::MathProgBase.SolverInterface.AbstractNLPEvaluator)

    @assert numConstr == 0  "OptimProblem can only solve unconstrained problems"
    @assert length(x_l) == length(x_u) "Lower and upper bounds have inconsistent dims"
    @assert all(x_l .<= x_u) "Lower and upper bounds have inconsistent dims"
    # if isa(m.s, Csminwel)
    #   @assert (all(!isfinite(x_l)) && all(!isfinite(x_l))) "Csminwel does not accept bounds on parameters"
    # end
    m.varLB = x_l
    m.varUB = x_u
    m.numVar = length(m.varLB)
    @assert m.numVar == length(m.varUB)
    m.initial_x = zeros(m.numVar)
    @assert sense == :Min || sense == :Max
    s = (sense == :Min) ? 1 : -1
    m.sense = s
    # Objective callback
    eval_f_cb(x) = s*MathProgBase.eval_f(d, x)
    # Objective gradient callback
    gr_cb(x, grad_f) = s*MathProgBase.eval_grad_f(d, grad_f, x)
    ad = any(map(x -> first(x), m.options) .== :autodiff)
    ng = :Grad ∈ MathProgBase.features_available(d)
    eval_grad_f_cb = getgradient(eval_f_cb, gr_cb, m.s, Val{ng}, Val{ad})
    m.inner = OptimProblem(nothing, eval_f_cb, eval_grad_f_cb, sense)
    m.d = d
    m.solve_time = 0.0
    m
end

## BFGS
getgradient(fcn, grad_f, s::Optimizer, ::Type{Val{true}}, ::Type{Val{false}}) = grad_f

function getgradient(fcn, grad_f, s::Optimizer, ::Type{Val{false}}, ::Type{Val{true}})
    gradient(x, gr) = ForwardDiff.gradient!(gr, fcn, x)
    gradient
end

function getgradient(fcn, grad_f, s::Optimizer, ::Type{Val{false}}, ::Type{Val{false}})
    gradient(x, gr) = gr[:] = Calculus.gradient(fcn, x)
    gradient
end


function unbounded(m::OptimMathProgModel)
    (all(MathProgBase.SolverInterface.getvarLB(m) == -Inf) && all(MathProgBase.SolverInterface.getvarUB(m) == +Inf)) ? true : false
end

function getoptimizer(s::Optim.Optimizer)
    notworking = (Newton, NewtonTrustRegion)
    for j in notworking
        isa(s, j) && error("Optimizer $j does not work with Fminbox")
    end
    m = (AcceleratedGradientDescent,
         BFGS,
         ConjugateGradient,
         GradientDescent,
         LBFGS,
         MomentumGradientDescent,
         NelderMead)
    out = BFGS
    for j in m
        isa(s, j) && (out = j; break)
    end
    out
end

function MathProgBase.SolverInterface.optimize!(m::OptimMathProgModel)
    if unbounded(m)
        out = Optim.optimize(m.inner.eval_f, m.inner.eval_grad_f, m.initial_x, m.s, Optim.Options(;m.options...))
    else
        method = getoptimizer(m.s)
        out = optimize(OnceDifferentiable(m.inner.eval_f, m.inner.eval_grad_f),
                 m.initial_x, MathProgBase.SolverInterface.getvarLB(m),
                 MathProgBase.SolverInterface.getvarUB(m), Fminbox(), optimizer = getoptimizer(m.s))
    end
    m.inner.out = out
end

MathProgBase.SolverInterface.getsense(m::OptimMathProgModel) = m.sense
MathProgBase.SolverInterface.numvar(m::OptimMathProgModel) = m.numVar
MathProgBase.SolverInterface.numconstr(m::OptimMathProgModel) = 0
MathProgBase.SolverInterface.status(m::OptimMathProgModel) = Optim.converged(m.inner.out)
MathProgBase.SolverInterface.getobjval(m::OptimMathProgModel) = minimum(m.inner.out)
MathProgBase.SolverInterface.getsolution(m::OptimMathProgModel) = minimizer(m.inner.out)
MathProgBase.SolverInterface.getvarLB(m::OptimMathProgModel) = m.varLB
MathProgBase.SolverInterface.getvarUB(m::OptimMathProgModel) = m.varUB
MathProgBase.SolverInterface.setvarLB!(m::OptimMathProgModel, x) = copy!(m.varLB, x)
MathProgBase.SolverInterface.setvarUB!(m::OptimMathProgModel, x) = copy!(m.varUB, x)
