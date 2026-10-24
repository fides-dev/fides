# Adversarial pass on the fides trust-region solver (2026-10-24)

Scripts in this folder reproduce everything below. Run them from the repo root with the
project venv (`python tests/adversarial/<script>.py`). `b1.log`, `b2.log`, `b3.log` are the
raw battery outputs (unconstrained / bounded / Hessian approximations; every subspace x
step-back combination plus `delta_init` and `theta_max` variations, with scipy baselines).

`harness.py` runs fides with a 30 s wall cap and 2000 iterations; "gap" is `f - f_opt`.

## Summary

| # | Finding | Severity | Where |
|---|---------|----------|-------|
| A1 | Hard case misclassified when `eigvals + lam == 0` but the gradient component is non-zero: `slam` keeps the undivided coefficient, so `secular(laminit)` is finite and the root search is skipped | **high** | `subproblem.py` `slam`/`solve_nd_trust_region_subproblem` |
| A2 | Hard-case step always adds `+sigma * v_min`; the sign that lowers the model is never chosen. Together with A1 this returns steps with **positive model value** (ascent) | **high** | `subproblem.py` hard case |
| A3 | Repeated eigenvalues tested with exact equality `(eigvals - mineig) == 0`; with `eig()` roundoff the "other" copy gets divided by ~1e-16 and the step is up to 1e15 x the radius | **high** | `subproblem.py` hard case |
| A4 | Acceptance tolerances are absolute (`delta + 1e-12`, `delta + sqrt(eps)`): valid roots are rejected for large `delta` (falls into hard case, 16x radius violation) and 10 % violations accepted for `delta ~ 1e-8` | medium | `subproblem.py` |
| A5 | Zero Hessian (or zero eigenvalue with gradient outside the null space): hard case returns an arbitrary eigenvector direction, model value > 0 | medium | `subproblem.py` |
| A6 | Newton on the secular equation uses absolute `tol=1e-12`; for `lambda >~ 1e4` it cannot converge, burns 100 iterations, then falls back to brentq | low (perf) | `subproblem.py` |
| B | Steihaug (`scg`) + any step-back: `step_back` scales `ss` but not `ss0`, while `s` already contains both. Predicted decrease, `nsx`, and reflected/truncated steps are built from inconsistent quantities; truncate can predict a decrease where the model actually increases | **high** for `scg` users | `steps.py` `TRStepSteihaug`, `Step.step_back` |
| C1 | FTOL fires on a step that was limited by the trust region: with `delta_init=1e-6` the solver stops after 1 iteration at a saddle (`||g||=2e-3`, min eigenvalue -1) and reports convergence | medium | `minimize.py` `check_convergence` |
| C2 | No first-order criterion for bound-constrained problems: GTOL uses the raw `||g||`, so at a bound-constrained optimum only FTOL (gated on `tr_ratio > mu`) or `DELTA_TOO_SMALL` can stop. When steps reach roundoff size the ratio is noise and the run ends with a rejection cascade and a *negative* exit flag at the optimum | medium | `minimize.py` |
| C3 | `tr_ratio` augmentation uses the new gradient; Coleman-Li's ratio uses the current one. At a zero-gradient start on a concave box it rejects a perfectly good first step | low | `minimize.py` `update_tr_radius` |
| D | 2D and `scg` subspaces are not affine invariant; with variable scales 1e-4..1e4 the 2D solver stops at f=9.2 and `scg` at f=534 (full subspace reaches the local minimum 3.99); partially A1 | medium | design |
| E | `BB` never converges on 2-D Rosenbrock (2000 iterations, f=2e-3); `DFP` needs 1368 | low (known weak updates) | `hessian_approximation.py` |

Candidate fixes for A1–A5 are monkeypatched in `validate_fix.py` (not applied to the package).
They turn the badly scaled 2-D Rosenbrock from "2000 iterations, f=3.57" into "25 iterations,
f=8e-16" and cut the subproblem fuzz failures from 208/880 to 42/880.

## A. Exact subproblem solver (`solve_nd_trust_region_subproblem`)

`fuzz_subproblem2.py` compares the solver against an eigendecomposition + bisection reference on
880 random instances (posdef, indefinite, ill-conditioned, hard, near-hard, repeated eigenvalues,
zero, semidefinite, huge, tiny; n = 2..30; delta = 1e-8..1e8; gradient scale 1e-8..1e8).
A case fails if the step is > 1e-6 outside the radius or the model value is > 1e-6 (relative)
above the reference. "Severe" means > 1 %.

| solver | failures | severe |
|--------|---------:|-------:|
| current | 208 / 880 | 85 |
| with candidate fixes | 42 / 880 | 15 (several look like reference noise at delta=1e8, g=1e-8) |

Minimal reproductions (all from the fuzz, see `validate_fix.py` header for the fixes):

```
# A3: B = Q diag(-1,-1) Q^T (random rotation), g ~ 1e-8, delta = 1
#     eig() returns -1 and -1 + 1e-16  ->  |s| = 1.7e7, model = -1.5e14   (optimum -0.5); 4 of 5 rotations fail
# A5: B = 0, g = [0.3, 0.4], delta = 1   ->  s = [1, 0], model = +0.30   (optimum -0.5)
# A4: posdef, eig in [1e-10, 1e6], delta = 1e8: Newton finds the root to 1e-24, but |s| exceeds delta by
#     1.6e-8 > sqrt(eps) -> rejected -> brentq -> rejected -> hard case -> |s| = 15.9 delta
```

### Why A1+A2 matter in practice (the badly scaled Rosenbrock)

`rosen2` with variables rescaled by (1e2, 1e-2): scipy `trust-exact` converges in 27 iterations;
fides (2D **and** full subspace) cycles accept/accept/reject forever and hits `maxiter=2000` at f=3.57.
Instrumenting the subproblem (`diag_secular.py`-style patch) shows the mechanism:

```
it 4  delta=8.0e+00  qpval=+1.609e-01  brute-force 2D optimum=-1.640e-01  chess_eig=[-4.8e-05, 7.5e+06]  cg=[+0.0203, 26.9]  sc=[+8.0, -3.6e-06]
```

The 2D subspace Hessian has a tiny negative eigenvalue (roundoff on a cond 1e11 matrix). `laminit = -mineig`,
`slam(laminit)` leaves the coefficient along `v_min` undivided, `secular(laminit)` comes out positive, the
Newton/brentq branch is skipped, the hard case zeroes `w_min` and adds `+delta * v_min`, i.e. a step
of length `delta` *up* the gradient (`cg[0] = +0.02`). Over the run, 669 of 2000 subproblem solves were
such misclassified hard cases, all 669 with positive model value. The radius update then shrinks the
radius and the cycle repeats. Also seen on `beale_far` (full: 33 misclassified, 10 ascent steps) and
`rosen10` (2D: 2 misclassified).

### A6 Newton tolerance

`diag_secular.py` counts fallbacks. With `quad20_c1e6` (2D) both Newton calls fail after 52 iterations
each (lambda ~ 4.5e3); with `rosen2` scaled by 1e10, 4 of 7 fail (lambda ~ 1e10-1e11). The fallback works,
so this is wasted work rather than wrong answers. The brentq bracket expansion (`xb *= 10`, 100 times,
shared `maxiter`) cannot reach lambda >~ 1e90; `expover3_far` (Hessian entries e^600) with the full
subspace ends in `DELTA_TOO_SMALL` at f=1e261 because of this plus A4 (2D subspace converges in 600
Newton-like iterations; scipy returns nan).

## B. Steihaug + step-back is internally inconsistent

`diag_qpval.py` recomputes the quadratic model at the step the optimizer actually takes
(`x_new - x`, in scaled coordinates) and compares to `step.qpval`. Every 2D/full combination is
consistent; `scg` is not as soon as a step-back is involved:

| problem | 2D reflect | full reflect | scg reflect | scg truncate | 2D truncate |
|---------|-----------:|-------------:|------------:|-------------:|------------:|
| boxquad10 (box-constrained quadratic) | 11 it | 11 it | 28 it, 20/28 steps inconsistent | 85 it, 54/85 inconsistent | 11 it |
| rosen10 in box [-2, 0.8] | 21 it | 22 it | 38 it, 3 inconsistent | 67 it, 10 inconsistent | 21 it |
| rosen2 in box | 20 it | 20 it | 48 it, DELTA_TOO_SMALL | 37 it | 19 it |

Example (boxquad10, scg truncate): `qpval = -2.300` but the model at the taken step is `+2.468`.
Root cause: `TRStepSteihaug.conj_grad` stores the CG prefix in `ss0` and the last leg in `ss`;
`Step.calculate` sets `s = scaling.dot(ss + ss0)` and then `step_back` multiplies `s`, `ss`, `sc` by
alpha but **not** `ss0`, so `qpval`, `ss + ss0` (used for `nsx`, `aug`, XTOL) and
`TRStepReflected`/`TRStepTruncated` (which reflect/truncate only `og_ss`, the last leg) all refer to a
different step than the one taken. Unconstrained `scg` is unaffected.

## C. Termination logic

**C1 FTOL on trust-region-limited steps.** `saddle_quadratic(10)` started at 1e-3 from the saddle:

| delta_init | iterations | exit | f | `||g||` at exit | min eig H |
|-----------:|-----------:|------|--:|-------:|------:|
| 1e-6 | 1 | FTOL | +3.8e-7 | 2.2e-3 | -1.00 |
| 1e-4 | 30 | FTOL | -33.75 | 1.2e-8 | +0.28 |
| 1 | 13 | FTOL | -31.09 | 2.1e-10 | +0.55 |

The first step is a full-radius step with ratio ~1 (the radius is about to double), but
`|f_new - f| = 1e-9 < fatol` so the run stops at a saddle and reports convergence. Same with
`concave_box10_zero`, `delta_init=1e-6`: 1 iteration, f=-1e-12 (optimum -40).

**C2 bound-constrained optima have no first-order exit.** At the solution of `min sum((x-3)^2)` on
`[-1,1]^n` every bound is active and `||g|| = 4 sqrt(n)`, so GTOL never fires and the only remaining
exits are FTOL (requires `tr_ratio > mu`) or `DELTA_TOO_SMALL`. Once steps are at roundoff size the
ratio is noise:

```
 7| +2.00E+01 | -1.1E-13 | -5.9E-01 | 2.0E+00 | 8.9E+00 | 6.2E-14 |  2d |0
 8| +2.00E+01 | +3.1E-13 | -2.6E+00 | 9.3E-08 | 8.9E+00 | 1.6E-14 |  2d |0
 ... 15 rejections ...
22| +2.00E+01 | +4.5E-13 | -8.6E+08 | 3.5E-16 | 8.9E+00 | 5.8E-23 |  2d |0
Stopping as trust region radius 8.69E-17 is smaller than machine precision.
```

Whether a run ends with FTOL (6 iterations) or DELTA_TOO_SMALL (22 iterations, exit flag -5) at the
*same* optimal point depends on the step-back strategy and `delta_init`: `allactive5`
(truncate/refine/delta_init=1e6), `allactive50` (truncate/refine), `start_at_lb5` (reflect, default),
`linear_box5` (9 of 13 settings), `quad5_tinybox` (all settings), `linear_box1` (2 settings). The scaled
gradient `v * g` is ~1e-13 in every case. Default `xtol=0` means XTOL can never help.

**C3 ratio augmentation uses the new gradient.** `update_tr_radius` computes
`aug = 0.5 * stepsx.dot(dv * |funout.grad| * stepsx)`; Coleman-Li (and `qpval`) use `|self.grad|`.
At `x0 = 0` on `-sum(x^2)` over `[-1, 2]^10` the first (correct, hard-case) step is rejected with ratio 0
because `g_old = 0` but `|g_new| = 2`. Swapping to the current gradient (`exp_aug.py` in the session)
changed iteration counts by 0-3 elsewhere, so this is minor.

## D. Scaling sensitivity of 2D / scg subspaces

`rosen10` with variables rescaled by `logspace(-k, k, 10)`; f_opt of the reachable basin is 3.987:

| scale span | 2D | full | scg |
|-----------:|----|------|-----|
| 1e-1..1e1 | 3.99 (38 it) | 3.99 (25) | 3.99 (60) |
| 1e-2..1e2 | 3.99 (53) | 3.99 (27) | 3.99 (172) |
| 1e-3..1e3 | 3.99 (52) | 3.99 (39) | **15.5 (977)** |
| 1e-4..1e4 | **9.24 (19)** | 3.99 (44) | **534 (509)** |

All failing runs exit with FTOL, i.e. they report convergence. For the 2-D case (scales 1e2/1e-2,
1e3/1e-3) both 2D and full hit `maxiter` because of A1/A2; with the candidate fixes they converge in
25-33 iterations.

## E. Hessian approximations (`b3.log`)

BFGS and SR1 behave; `BB` reaches `maxiter=2000` on *every* problem including 2-D Rosenbrock
(f=1.8e-3) and `DFP` needs 1368 iterations on 2-D Rosenbrock and does not converge on `rosen10/30`.
These are known-weak updates but the docs do not warn about them.

## Not bugs (for the record)

- Extended Rosenbrock for n >= 4 has a local minimum at f=3.9866; every method (scipy included) finds
  it from the standard start. `cosine_mixture`, `concave_box`, `saddle10` likewise have many local minima.
- 10x underestimated Hessian: ~linear convergence, `maxiter` for fides and scipy alike.
- `exp(x)` from x=600: 600 Newton-like iterations (scipy `trust-exact` returns nan).
- `rosen2 * 1e-10` stops after 1 iteration: absolute `fatol=1e-8` is larger than the objective.
- Variable scales 1e-6/1e6 (Hessian condition ~1e24) are beyond double precision for everyone.
- 2D step cost: `TRStep2D` does a dense `eig` + `lstsq` per iteration (n=100: ~10 ms/it, 7x slower than scg).
