#!/usr/bin/env python
"""Simple test to understand dynesty validity"""

import numpy as np
import dynesty

# Simple test functions
def simple_prior_transform(u):
    # Transform unit cube to [-5, 5] for each parameter
    return u * 10 - 5

def simple_loglike(theta):
    # Simple Gaussian likelihood centered at origin
    return -0.5 * np.sum(theta**2)

# Test if this works with dynesty
print("Testing simple case...")
sampler = dynesty.NestedSampler(simple_loglike, simple_prior_transform, 2, nlive=50)
try:
    sampler.run_nested(maxiter=100)
    print("Simple case worked!")
except Exception as e:
    print(f"Simple case failed: {e}")

# Now test with very negative log-likelihoods like ours
def negative_loglike(theta):
    return -25515.0 - 0.5 * np.sum(theta**2)

print("\nTesting very negative log-likelihood...")
sampler2 = dynesty.NestedSampler(negative_loglike, simple_prior_transform, 2, nlive=50)
try:
    sampler2.run_nested(maxiter=100)
    print("Negative case worked!")
except Exception as e:
    print(f"Negative case failed: {e}")

# Test with our actual functions but simplified
print("\nTesting with our functions...")
import sys
sys.path.insert(0, '.')

from Data import Data
from Event import Event
from Fit import Fit
from Orbit import Orbit
from Parallax import Parallax

# Set up like our debug script but simplified
LOM_enabled = False
ndim = 2  # Just test with 2 parameters
labels = ["s", "q"]

orbit_obj = Orbit()
fit_obj = Fit(sampling_package="dynesty", LOM_enabled=LOM_enabled, ndim=ndim, labels=labels)
data_obj = Data()

path = "overguide_m00_Fisher/"
event_name, truths_series, data = data_obj.new_event(path, sort="alphanumeric")
truths = truths_series.to_dict()

if 'params' in truths and isinstance(truths['params'], list):
    truths['params'] = np.array(truths['params'])

# Create event objects
piE = np.array([truths["piEN"], truths["piEE"]])
tu_data, epochs = {}, {}
for obs in data.keys():
    tu_data[obs] = data[obs][3:5, :].T
    epochs[obs] = data[obs][0, :]

parallax_obj = Parallax(truths["ra_deg"], truths["dec_deg"], orbit_obj, truths["tcroin"], tu_data, piE, epochs)
parallax_obj.update_piE_NE(truths["piEN"], truths["piEE"])
event_fit = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, truths["t0lens1"], LOM_enabled=LOM_enabled)

# Simple priors for just 2 parameters
p_unc_log_space = np.array([0.1, 0.1])
prange_log = p_unc_log_space * 2.0
prange_linear = np.array([])
normal = True

fit_obj.current_event = event_fit

def our_ptransform(u):
    return fit_obj.prior_transform(u, truths["params"][:ndim], prange_linear, prange_log, normal, None)

def our_loglike(u):
    return fit_obj.lnprob_transform(u, event_fit, truths, prange_linear, prange_log, normal, None)

# Test a few points manually
print("Testing our functions manually...")
for i in range(5):
    u_test = np.random.rand(ndim)
    try:
        theta_test = our_ptransform(u_test)
        ll_test = our_loglike(u_test)
        print(f"u={u_test} -> theta={theta_test} -> loglike={ll_test}")
        if not np.isfinite(ll_test):
            print(f"  Non-finite log-likelihood!")
    except Exception as e:
        print(f"  Error: {e}")

print("\nTesting with dynesty...")
try:
    sampler3 = dynesty.NestedSampler(our_loglike, our_ptransform, ndim, nlive=20)
    sampler3.run_nested(maxiter=50)
    print("Our case worked!")
except Exception as e:
    print(f"Our case failed: {e}") 