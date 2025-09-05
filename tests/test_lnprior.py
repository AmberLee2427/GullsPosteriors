import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Fit import Fit


def test_negative_q_returns_neg_inf():
    fit = Fit(sampling_package="emcee", LOM_enabled=True)
    theta = [
        1.0,   # s
        -0.5,  # q (negative to trigger -inf prior)
        0.001, # rho
        0.1,   # u0
        1.0,   # alpha
        0.0,   # t0
        30.0,  # tE
        0.0,   # piEE
        0.0,   # piEN
        0.1,   # i
        0.0,   # phase
        150.0, # period (> 4 * tE)
    ]
    assert fit.lnprior(theta) == -np.inf

