#!/usr/bin/env python
"""Debug script to test dynesty setup"""

import numpy as np
import sys
import os

# Add the current directory to path so we can import modules
sys.path.insert(0, '.')

from Data import Data
from Event import Event
from Fit import Fit
from Orbit import Orbit
from Parallax import Parallax

# Test with a simple case
LOM_enabled = False
ndim = 9
labels = ["s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN"]

# Create objects
orbit_obj = Orbit()
fit_obj = Fit(sampling_package="dynesty", LOM_enabled=LOM_enabled, ndim=ndim, labels=labels)
data_obj = Data()

# Get one event
path = "overguide_m00_Fisher/"
event_name, truths_series, data = data_obj.new_event(path, sort="alphanumeric")
truths = truths_series.to_dict()

# Ensure params is numpy array
if 'params' in truths and isinstance(truths['params'], list):
    truths['params'] = np.array(truths['params'])

# Create parallax object like in main script
piE = np.array([truths["piEN"], truths["piEE"]])
t0 = truths["params"][5]
tE = truths["params"][6]
tu_data, epochs, t_data, f_true, f_err_true = {}, {}, {}, {}, {}
for obs in data.keys():
    tu_data[obs] = data[obs][3:5, :].T
    epochs[obs] = data[obs][0, :]
    f_true[obs] = data[obs][5, :]
    f_err_true[obs] = data[obs][6, :]
    t_data[obs] = data[obs][0, :]

parallax_obj = Parallax(
    truths["ra_deg"],
    truths["dec_deg"],
    orbit_obj,
    truths["tcroin"],
    tu_data,
    piE,
    epochs,
)
parallax_obj.update_piE_NE(truths["piEN"], truths["piEE"])

# Create event
event_fit = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, truths["t0lens1"], LOM_enabled=LOM_enabled)

# Set up priors
p_unc_log_space = np.array([0.1, 0.1, 0.1])
prange_log = p_unc_log_space * 2.0
linear_indices = [3, 4, 5, 6, 7, 8]
p_unc = np.array([0.05, 0.05, 0.1, 0.1, 0.05, 0.5, 2.5, 2.5, 5.0])
p_unc_linear_space = p_unc[linear_indices]
prange_linear = p_unc_linear_space * 2.0
normal = True

print("Testing prior transform...")
u_test = np.ones(ndim) * 0.5
theta_test = fit_obj.prior_transform(u_test, truths["params"][:ndim], prange_linear, prange_log, normal, None)
print(f"u_test: {u_test}")
print(f"theta_test: {theta_test}")

print("\nTesting likelihood...")
fit_obj.current_event = event_fit
lp_test = fit_obj.lnprob(theta_test, event_fit)
print(f"lnprob: {lp_test}")

print("\nTesting lnprob_transform...")
lp_transform_test = fit_obj.lnprob_transform(u_test, event_fit, truths, prange_linear, prange_log, normal, None)
print(f"lnprob_transform: {lp_transform_test}")

print("\nTesting random unit cube points...")
for i in range(10):
    u_random = np.random.rand(ndim)
    try:
        theta_random = fit_obj.prior_transform(u_random, truths["params"][:ndim], prange_linear, prange_log, normal, None)
        lp_random = fit_obj.lnprob(theta_random, event_fit)
        print(f"u_{i}: {u_random[:3]}... -> lnprob: {lp_random}")
    except Exception as e:
        print(f"u_{i}: {u_random[:3]}... -> ERROR: {e}") 