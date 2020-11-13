# Fides - A python package for Trust Region Optimization

<a href="https://badge.fury.io/py/fides">
  <img src="https://badge.fury.io/py/fides.svg" alt="PyPI version"></a>
<a href="https://codecov.io/gh/fides-dev/fides">
  <img src="https://codecov.io/gh/fides-dev/fides/branch/master/graph/badge.svg" alt="Code coverage"></a>
<a href="https://fides-optimizer.readthedocs.io/en/latest/?badge=latest">
 <img src="https://readthedocs.org/projects/fides-optimizer/badge/?version=latest" alt="ReadTheDocs status"></a>
<a href="https://zenodo.org/badge/latestdoi/312057973">
 <img src="https://zenodo.org/badge/312057973.svg" alt="DOI"></a>

## About Fides

Fides implements an Interior Trust Region Reflective for boundary costrained
optimization problems based on the papers [ColemanLi1994] and [ColemanLi1996
]. Accordingly, Fides is named after the Roman goddess of trust and
reliability. In contrast to other optimizers, Fides solves the full trust
-region subproblem exactly, which can yields higher quality proposal steps, but
is computationally more expensive. This makes Fides particularly attractive
for optimization problems with objective functions that are computationally
expensive to evaluate and the computational cost of solving the trust
 -region subproblem is negligible.

Fides can be installed via `pip install fides`. Further documentation is
 avaliable at [Read the Docs](fides-optimizer.readthedocs.io).
 
 
## Features


* Boundary constrained interior trust-region optimization
* Recursive Reflective and Truncated constraint management
* Full and 2D subproblem solution solvers
* BFGS, DFP and SR1 Hessian Approximations

