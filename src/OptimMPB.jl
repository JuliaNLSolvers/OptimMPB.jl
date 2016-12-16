module OptimMPB

using Reexport

using ForwardDiff
using Calculus
@reexport using Optim
@reexport using MathProgBase
import MathProgBase: NonlinearModel, getsense, numvar, numconstr, SolverInterface.optimize!
                     MathProgBase.SolverInterface.loadproblem!
import Optim: Optimizer, OptimizationResults, optimize, MultivariateOptimizationResults,
              initial_state, minimizer, minimum, iterations, converged, x_converged,
              f_converged, f_tol, g_tol, g_converged, iteration_limit_reached, f_calls, method

include("interface.jl")

export OptimSolver

end # module
