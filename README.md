# OptimMPB.jl

[![Build Status](https://travis-ci.org/gragusa/OptimMPB.jl.svg?branch=master)](https://travis-ci.org/gragusa/OptimMPB.jl) [![Coverage Status](https://coveralls.io/repos/gragusa/OptimMPB.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/OptimMPB.jl?branch=master) [![codecov.io](http://codecov.io/github/gragusa/OptimMPB.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/OptimMPB.jl?branch=master)

This package provides glue code for using algorithms in `Optim.jl` using the `MathProgBase` interface. This is still a work in progress and ultimately the code here should belong to `Optim.jl`.

This simple example illustrate how things work:

```julia
using OptimMPB  ## OptimMPB reexport Optim and MathProgBase

type Rosenbrock <: MathProgBase.AbstractNLPEvaluator
end

MathProgBase.features_available(d::Rosenbrock) = [:Grad]
MathProgBase.eval_f(d::Rosenbrock, x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
function MathProgBase.eval_grad_f(d::Rosenbrock, gr, x)  
  gr[1] = -2*(1-x[1]) - 400*x[1]*(x[2]-x[1]^2)
  gr[2] = 200*(x[2]-x[1]^2)
end

function MathProgBase.jac_structure(d::Rosenbrock)
    Int[],Int[]
end

function MathProgBase.eval_jac_g(d::Rosenbrock, J, x)
    nothing
end


s = OptimSolver(BFGS)
m = MathProgBase.NonlinearModel(s)
MathProgBase.loadproblem!(m, 2, [-1,-1], [2.,2.], :Min, Rosenbrock())
MathProgBase.setwarmstart!(m, [0.1,-0.1])
MathProgBase.optimize!(m)

MathpogBase.getobjval(m)
MathProgBase.getsolution(m)

```

If `MathProgBase.features_available(d::AbstractNLPEvaluator) = []` then numerical derivatives are used to obtain gradient information.
