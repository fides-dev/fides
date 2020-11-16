from fides import BFGS

import pytest
import numpy as np


def test_wrong_dim():
    with pytest.raises(ValueError):
        BFGS(hess_init=1)

    with pytest.raises(ValueError):
        BFGS(hess_init=np.ones(2,))

    with pytest.raises(ValueError):
        BFGS(hess_init=np.ones((2, 3, 2)))

    with pytest.raises(ValueError):
        BFGS(hess_init=np.ones((2, 3)))

    with pytest.raises(ValueError):
        h = BFGS(hess_init=np.ones((2, 2)))
        h.init_mat(3)
