using OptimMPB
using Base.Test

# write your own tests here
type EX1 <: MathProgBase.AbstractNLPEvaluator end


MathProgBase.features_available(d::EX1) = [:Grad]
MathProgBase.eval_f(d::EX1, x) = (309.0 - 5.0 * x[1])^2 + (17.0 - x[2])^2
function MathProgBase.eval_grad_f(d::EX1, gr, x)
  gr[1] = -10.0 * (309.0 - 5.0 * x[1])
  gr[2] = -2.0 * (17.0 - x[2])
end

function MathProgBase.jac_structure(d::EX1)
    Int[],Int[]
end

function MathProgBase.eval_jac_g(d::EX1, J, x)
    nothing
end

for method in (BFGS, LBFGS, GradientDescent)
  s = OptimSolver(BFGS)
  m = MathProgBase.NonlinearModel(s)
  MathProgBase.loadproblem!(m, 2, [-1.,-1.], [2.,2.], :Min, EX1())
  MathProgBase.setwarmstart!(m, [0.1,-0.1])
  MathProgBase.optimize!(m)

  @test Optim.minimizer(m.inner.out) == MathProgBase.getsolution(m)
  @test Optim.minimum(m.inner.out) == MathProgBase.getobjval(m)
  @test MathProgBase.getvarLB(m) == [-1., -1]
  @test MathProgBase.getvarUB(m) == [2., 2]
  @test MathProgBase.status(m) == Optim.converged(m.inner.out)
  @test norm(Optim.minimizer(m.inner.out) - [309.0 / 5.0, 17.0]) < 0.01
  @test Optim.g_converged(m.inner.out)
end

for method in (BFGS, LBFGS, GradientDescent)
  s = OptimSolver(BFGS, show_trace = true, store_trace = true, extended_trace = true)
  m = MathProgBase.NonlinearModel(s)
  MathProgBase.loadproblem!(m, 2, [-1.,-1.], [2.,2.], :Min, EX1())
  MathProgBase.setwarmstart!(m, [0.1,-0.1])
  MathProgBase.optimize!(m)

  @test Optim.minimizer(m.inner.out) == MathProgBase.getsolution(m)
  @test Optim.minimum(m.inner.out) == MathProgBase.getobjval(m)
  @test MathProgBase.getvarLB(m) == [-1., -1]
  @test MathProgBase.getvarUB(m) == [2., 2]
  @test MathProgBase.status(m) == Optim.converged(m.inner.out)
  @test norm(Optim.minimizer(m.inner.out) - [309.0 / 5.0, 17.0]) < 0.01
  @test Optim.g_converged(m.inner.out)
end

## CSMINWEL
s = OptimSolver(Csminwel, show_trace = true, store_trace = true, extended_trace = true)
m = MathProgBase.NonlinearModel(s)
MathProgBase.loadproblem!(m, 2, [-Inf,-Inf], [+Inf,+Inf], :Min, EX1())
MathProgBase.setwarmstart!(m, [0.1,-0.1])
MathProgBase.optimize!(m)
@test Optim.minimizer(m.inner.out) == MathProgBase.getsolution(m)
@test Optim.minimum(m.inner.out) == MathProgBase.getobjval(m)
@test MathProgBase.getvarLB(m) == [-Inf, -Inf]
@test MathProgBase.getvarUB(m) == [+Inf, +Inf]
@test MathProgBase.status(m) == Optim.converged(m.inner.out)
@test norm(Optim.minimizer(m.inner.out) - [309.0 / 5.0, 17.0]) < 0.01
@test Optim.g_converged(m.inner.out)
