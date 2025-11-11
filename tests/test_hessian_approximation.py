import numpy as np
import pytest

from fides import BFGS, Broyden


def test_wrong_dim():
    h = BFGS(init_with_hess=True)
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
