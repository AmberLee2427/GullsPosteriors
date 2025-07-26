import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the test
center = 0.0  # center at 0 for the new mapping
max_sigma = (np.pi - 1e-6) / 8
sigma = max_sigma
loc = center
scale = sigma

# Range of theta (physical values)
theta_vals = np.linspace(center - 4 * scale, center + 4 * scale, 500)
u_vals = np.linspace(0, 1, 500)

# Compute CDF and PPF
cdf_vals = norm.cdf(theta_vals, loc=loc, scale=scale)
ppf_vals = norm.ppf(u_vals, loc=loc, scale=scale)

# Compute theta for u=0.25, 0.5, 0.75
theta_025 = norm.ppf(0.25, loc=loc, scale=scale)
theta_05 = norm.ppf(0.5, loc=loc, scale=scale)
theta_075 = norm.ppf(0.75, loc=loc, scale=scale)

# Plot CDF: theta -> u
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(theta_vals, cdf_vals, label="norm.cdf(theta)")
plt.axvline(center, color="k", linestyle="--", label="center (0)")
plt.axhline(0.25, color="r", linestyle=":", label="u=0.25/0.75")
plt.axhline(0.5, color="g", linestyle=":", label="u=0.5")
plt.axhline(0.75, color="r", linestyle=":")
plt.xlabel("theta (radians)")
plt.ylabel("u (unit cube)")
plt.title("CDF: theta → u")
plt.legend()

# Plot PPF: u -> theta
plt.subplot(1, 2, 2)
plt.plot(u_vals, ppf_vals, label="norm.ppf(u)")
plt.axvline(0.25, color="r", linestyle=":", label="u=0.25/0.75")
plt.axvline(0.5, color="g", linestyle=":", label="u=0.5")
plt.axvline(0.75, color="r", linestyle=":")
plt.axhline(center, color="k", linestyle="--", label="center (0)")
plt.axhline(theta_025, color="r", linestyle="--", label="theta@u=0.25/0.75")
plt.axhline(theta_05, color="g", linestyle="--", label="theta@u=0.5")
plt.axhline(theta_075, color="r", linestyle="--")
plt.xlabel("u (unit cube)")
plt.ylabel("theta (radians)")
plt.title("PPF: u → theta")
plt.legend()

plt.tight_layout()
plt.savefig("tests/alpha_cdf_plot_maxsigma.png")
plt.show() 