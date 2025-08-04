#!/usr/bin/env python
"""
Quick optimisation + Laplace covariance for a single event folder passed as arg.
Saves  <event_dir>/laplace_cov.npy   and prints a 'score' versus Fisher.
"""
import numpy as np, sys, json
from pathlib import Path
from scipy.optimize import minimize
from Fit import Fit
from Data import Data
from Event import Event
from Orbit import Orbit
from VBMicrolensing import VBMicrolensing   # only if needed
import time

start_time = time.time()
# ----------------------------------------------------------------------
event_path = Path(sys.argv[1])
assert event_path.is_dir(), "Need event directory"

# -- rebuild exactly the same objects gulls_post_emcee_adaptive_BI uses ----
# Here we only need Fit.lnprob(theta, event)
orbit_obj  = Orbit()
fit_obj    = Fit(sampling_package="emcee", LOM_enabled=False, ndim=9,
                 labels=["s","q","rho","u0","alpha","t0","tE","piEE","piEN"])
data_obj   = Data()
event_name, truths_ser, data = data_obj.new_event(event_path.as_posix()+"/",
                                                  sort="alphanumeric")
truths = truths_ser.to_dict()
parallax_obj = None        # build if your Fit.lnprob needs it
event = Event(parallax_obj, orbit_obj, data, truths,
              data_obj.sim_time0, truths["t0lens1"], LOM_enabled=False)

theta0 = np.array(truths["params"][:9])
def nll(theta):            # negative log-likelihood
    return -fit_obj.lnprob(theta, event)

res = minimize(nll, theta0, method="Nelder-Mead",
               options=dict(maxiter=3000))
theta_map = res.x
print("MAP found:", theta_map)

# Finite-difference Hessian
eps = 1e-4
nd  = len(theta_map)
H = np.zeros((nd, nd))
for i in range(nd):
    ei = np.zeros(nd); ei[i] = eps
    for j in range(i, nd):
        ej = np.zeros(nd); ej[j] = eps
        f1 = nll(theta_map + ei + ej)
        f2 = nll(theta_map + ei - ej)
        f3 = nll(theta_map - ei + ej)
        f4 = nll(theta_map - ei - ej)
        H[i,j] = H[j,i] = (f1 - f2 - f3 + f4)/(4*eps*eps)
# Regularise & invert
lam = 1e-3*np.mean(np.diag(H))
Sigma = np.linalg.inv(H + lam*np.eye(nd))
np.save(event_path/"laplace_cov.npy", Sigma)
print("Laplace covariance saved.")

print("Time taken:", time.time() - start_time)

# optional: compare to Fisher using scripts.check_fisher_permutation machinery
