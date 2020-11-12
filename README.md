# Fides - A python package for Trust Region Optimization

<a href="https://badge.fury.io/py/fides">
  <img src="https://badge.fury.io/py/fides.svg" alt="PyPI version"></a>
<a href="https://codecov.io/gh/Fides-dev/fides">
  <img src="https://codecov.io/gh/ides-dev/fides/branch/master/graph/badge.svg" alt="Code coverage"></a>
<a href="https://fides-optimizer.readthedocs.io/en/latest/?badge=latest">
 <img src="https://readthedocs.org/projects/fides-optimizer/badge/?version=latest" alt="ReadTheDocs status"></a>


Fides implements an Interior Trust Region Reflective optimizer based on
the papers [ColemanLi1994] and [ColemanLi1996]. In contrast to other
optimizers, the trust-region subproblem can be solved exactly, which
yields higher quality proposal steps, but is computationally more expensive.
This makes Fides particularly attractive for optimization problems with
objective functions that are computationally expensive to evaluate and the
computational cost of solving the trust-region subproblem is negligible.

To emphasize this ideal of reliably computing good proposal steps, Fides is
named after the Roman goddess of trust and reliability.