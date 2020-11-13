===========
About Fides
===========

Fides implements an Interior Trust Region Reflective for boundary costrained
optimization problems based on the papers [ColemanLi1994] and [ColemanLi1996
]. Accordingly, Fides is named after the Roman goddess of trust and
reliability. In contrast to other optimizers, Fides solves the full trust
-region subproblem exactly, which can yields higher quality proposal steps, but
is computationally more expensive. This makes Fides particularly attractive
for optimization problems with objective functions that are computationally
expensive to evaluate and the computational cost of solving the trust
-region subproblem is negligible.

Features
========

* Boundary constrained interior trust-region optimization
* Recursive Reflective and Truncated constraint management
* Full and 2D subproblem solution solvers
* BFGS, DFP and SR1 Hessian Approximations

