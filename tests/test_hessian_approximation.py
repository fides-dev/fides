import numpy as np
import pytest

from fides import BFGS, Broyden


def test_wrong_dim():
    with pytest.raises(ValueError):
        h = BFGS(init_with_hess=True)
        h.init_mat(dim=3, hess=np.ones((2, 2)))

    with pytest.raises(ValueError):
        h = BFGS()
        h.init_mat(dim=3)
        h.set_mat(np.ones((2, 2)))


def test_broyden():
    h = Broyden(phi=2)
    h.init_mat(dim=2)
    h.update(np.random.random((2, 1)), np.random.random((2, 1)))

    h = Broyden(phi=-1)
    h.init_mat(dim=2)
    h.update(np.random.random((2,)), np.random.random((2,)))
