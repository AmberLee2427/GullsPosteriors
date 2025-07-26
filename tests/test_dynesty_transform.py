"""
Test round-trip of prior_transform and detransform_theta in Fit.

Before running, activate the conda environment:
    conda activate GullsPosteriors

Run with:
    pytest tests/test_dynesty_transform.py
or
    python tests/test_dynesty_transform.py
"""
import numpy as np
import pytest
from Fit import Fit

# Example parameter setup (9D, no LOM)
labels = ["s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN"]
ndim = len(labels)

# Truth values (arbitrary but reasonable)
truths = np.array([1.2, 0.01, 0.001, 0.1, 1.0, 2450000.0, 30.0, 0.0, 0.0])
prange_log = np.array([1.0, 2.0, 2.0])  # log10 width for s, q, rho (e.g., 10^(log_true +/- 0.5))
prange_linear = np.array([0.2, np.pi, 100.0, 100.0, 100.0, 100.0])  # for u0, alpha, t0, tE, piEE, piEN

# Define realistic 1-sigma Fisher uncertainties for testing
# These should be small, representative uncertainties, not prior widths.
# Adjust these values based on typical uncertainties for your parameters.
# For alpha, a small value like 0.1 radians is more typical for 1-sigma.
test_fisher_uncertainties = np.array([
    0.1,   # s (example 1-sigma)
    0.001, # q (example 1-sigma)
    0.0001,# rho (example 1-sigma)
    0.01,  # u0 (example 1-sigma)
    0.1,   # alpha (example 1-sigma, in radians, much smaller than 2*pi)
    1.0,   # t0 (example 1-sigma)
    0.5,   # tE (example 1-sigma)
    0.01,  # piEE (example 1-sigma)
    0.01   # piEN (example 1-sigma)
])


@pytest.mark.parametrize("u", [
    np.full(ndim, 0.5),
    np.full(ndim, 0.25),
    np.full(ndim, 0.75),
    np.random.rand(ndim),
])
def test_round_trip(u):
    fit = Fit(sampling_package="dynesty", LOM_enabled=False, ndim=ndim, labels=labels)
    theta = fit.prior_transform(u, truths, prange_linear, prange_log, normal=False, fisher_uncertainties=None) # Ensure Fisher is None for this test
    # 0.5 in unit cube should map to truth
    if np.allclose(u, 0.5):
        assert np.allclose(theta, truths, rtol=1e-10, atol=1e-10), f"0.5 in unit cube should map to truths: {theta} vs {truths}"
    # Round-trip: u -> theta -> u2 should be close to u
    u2 = fit.detransform_theta(theta, truths, prange_linear, prange_log, normal=False, fisher_uncertainties=None) # Ensure Fisher is None for this test
    assert np.allclose(u, u2, rtol=1e-8, atol=1e-8), f"Round-trip failed: u={u}, u2={u2}"

@pytest.mark.parametrize("u", [
    np.full(ndim, 0.5),
    np.full(ndim, 0.25),
    np.full(ndim, 0.75),
    np.random.rand(ndim),
])
def test_round_trip_fisher(u):
    fit = Fit(sampling_package="dynesty", LOM_enabled=False, ndim=ndim, labels=labels)
    
    # Use the newly defined test_fisher_uncertainties
    fisher_uncertainties = test_fisher_uncertainties 

    theta = fit.prior_transform(u, truths, prange_linear, prange_log, normal=False, fisher_uncertainties=fisher_uncertainties)
    # 0.5 in unit cube should map to truth
    if np.allclose(u, 0.5):
        assert np.allclose(theta, truths, rtol=1e-10, atol=1e-10), f"0.5 in unit cube should map to truths (Fisher): {theta} vs {truths}"
    # Round-trip: u -> theta -> u2 should be close to u
    u2 = fit.detransform_theta(theta, truths, prange_linear, prange_log, normal=False, fisher_uncertainties=fisher_uncertainties)
    assert np.allclose(u, u2, rtol=1e-8, atol=1e-8), f"Round-trip failed (Fisher): u={u}, u2={u2}"

if __name__ == "__main__":
    # Run all tests manually
    for u in [
        np.full(ndim, 0.5),
        np.full(ndim, 0.25),
        np.full(ndim, 0.75),
        np.random.rand(ndim),
    ]:
        print(f"Testing u={u}")
        test_round_trip(u)
        test_round_trip_fisher(u)
    print("All manual tests passed.")
