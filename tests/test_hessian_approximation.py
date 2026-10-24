import numpy as np
import pytest

from fides import BB, BFGS, BG, DFP, SR1, Broyden, HybridFixed, HybridFraction


def test_wrong_dim():
    h = BFGS()
    with pytest.raises(ValueError):
        h._init_mat(dim=3, hess=np.ones((2, 2)))

    h = BFGS()
    h._init_mat(dim=3)
    with pytest.raises(ValueError):
        h.set_mat(np.ones((2, 2)))


def test_broyden():
    h = Broyden(phi=2)
    h._init_mat(dim=2)
    h.update(np.random.random((2, 1)), np.random.random((2, 1)))

    h = Broyden(phi=-1)
    h._init_mat(dim=2)
    h.update(np.random.random((2,)), np.random.random((2,)))


ITERATIVE_SCHEMES = [BFGS, DFP, SR1, BG, BB, lambda: Broyden(phi=0.5)]


def _random_pair(rng, dim):
    s = rng.normal(size=dim)
    y = s + 0.3 * rng.normal(size=dim)
    if y.dot(s) <= 0:
        y = -y
    return s, y


@pytest.mark.parametrize('happ', ITERATIVE_SCHEMES)
def test_secant_equation(happ):
    rng = np.random.default_rng(0)
    dim = 4
    h = happ()
    h._init_mat(dim)
    for _ in range(5):
        s, y = _random_pair(rng, dim)
        hess_before = h._hess.copy()
        h.update(s, y)
        # update must actually be applied
        assert not np.allclose(h._hess, hess_before)
        np.testing.assert_allclose(h._hess.dot(s), y, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('happ', [BG, BB])
def test_nonsymmetric_schemes_return_symmetric_part(happ):
    rng = np.random.default_rng(1)
    dim = 3
    h = happ()
    h._init_mat(dim)
    for _ in range(3):
        h.update(*_random_pair(rng, dim))
    # internal approximation is not symmetric, but what the optimizer
    # sees is the symmetric part
    assert not np.allclose(h._hess, h._hess.T)
    mat = h.get_mat()
    np.testing.assert_array_equal(mat, mat.T)
    np.testing.assert_allclose(mat, (h._hess + h._hess.T) / 2)
    diff = h.get_diff()
    np.testing.assert_array_equal(diff, diff.T)


@pytest.mark.parametrize('happ', ITERATIVE_SCHEMES)
def test_integer_init(happ):
    h = happ()
    h._init_mat(dim=2, hess=np.eye(2, dtype=int))
    assert h.get_mat().dtype == float
    h.update(np.array([1.0, 0.5]), np.array([2.0, 0.3]))


@pytest.mark.parametrize(
    ('happ', 'update_args'),
    [
        (lambda: HybridFixed(BFGS(), switch_iteration=1), (1,)),
        (lambda: HybridFraction(BFGS(), switch_threshold=0.5), (30, 30)),
    ],
)
def test_hybrid_switch_reset(happ, update_args):
    h = happ()
    assert h.requires_hess is True
    h._init_mat(dim=2)
    assert not h._switched
    s, y = np.array([1.0, 0.5]), np.array([2.0, 0.3])
    h.update(s, y, np.eye(2), *update_args)
    assert h._switched
    # a new optimization run must start with the dynamic approximation
    h._init_mat(dim=2)
    assert not h._switched


def test_set_mat_casts_to_float():
    h = BFGS()
    h._init_mat(dim=2)
    h.set_mat(np.eye(2, dtype=int) * 2)
    assert h.get_mat().dtype == float
    h.update(np.array([1.0, 0.5]), np.array([2.0, 0.3]))


def test_bb_degenerate_update_is_skipped():
    h = BB()
    h._init_mat(dim=2)
    # y^T B s == 0, the update is undefined and must be skipped
    h.update(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    np.testing.assert_array_equal(h.get_diff(), np.zeros((2, 2)))
    np.testing.assert_array_equal(h.get_mat(), np.eye(2))


@pytest.mark.parametrize('happ', [BFGS, DFP, lambda: Broyden(phi=0.5)])
def test_broyden_zero_initialization(happ):
    """
    a zero initial approximation must not produce nan (0/0) updates, the
    update reduces to the rank-1 term y y^T / (y^T s)
    """
    h = happ()
    h._init_mat(dim=2, hess=np.zeros((2, 2)))
    s = np.array([1.0, 0.5])
    y = np.array([2.0, 0.3])
    h.update(s, y)
    assert np.isfinite(h.get_mat()).all()
    np.testing.assert_allclose(h.get_mat(), np.outer(y, y) / y.dot(s))
    np.testing.assert_allclose(h.get_mat().dot(s), y)


def test_broyden_zero_curvature_update_is_skipped():
    h = BFGS(enforce_curv_cond=False)
    h._init_mat(dim=2)
    # y orthogonal to s, division by y^T s is undefined
    h.update(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.isfinite(h.get_mat()).all()
    np.testing.assert_array_equal(h.get_diff(), np.zeros((2, 2)))

    # s^T B s == 0 but B s != 0, division by s^T B s is undefined
    h = BFGS()
    h._init_mat(dim=2, hess=np.array([[0.0, 1.0], [1.0, 0.0]]))
    h.update(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    assert np.isfinite(h.get_mat()).all()
    np.testing.assert_array_equal(h.get_diff(), np.zeros((2, 2)))


def test_init_mat_rejects_non_square():
    h = BFGS()
    with pytest.raises(ValueError, match='inconsistent shape'):
        h._init_mat(dim=2, hess=np.ones((2, 3)))


def test_sr1_skips_update_if_secant_equation_holds():
    h = SR1()
    h._init_mat(dim=2)
    s = np.array([1.0, 0.5])
    # y == B s, the update would be 0/0
    h.update(s, h.get_mat().dot(s))
    assert np.isfinite(h.get_mat()).all()
    np.testing.assert_array_equal(h.get_diff(), np.zeros((2, 2)))
