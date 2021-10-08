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

Fides implements an Interior Trust Region Reflective for boundary constrained
optimization problems based on the papers
[ColemanLi1994](https://doi.org/10.1007/BF01582221) and
[ColemanLi1996](http://dx.doi.org/10.1137/0806023). Accordingly, Fides is named
after the Roman goddess of trust and
reliability.

Fides can be installed via `pip install fides`. Further documentation is
 available at [Read the Docs](https://fides-optimizer.readthedocs.io/).
 
 
## Features


* Boundary constrained and unconstrained interior trust-region optimization
* Reflective, truncated and optimization based boundary heuristics
* Exact, 2D and CG subproblem solvers
* BFGS, DFP, SR1, PSB, Broyden (good and bad) and Broyden class iterative
 Hessian Approximation schemes
* SSM, TSSM, FX, GNSBFGS and custom hybrid Hessian Approximations schemes

