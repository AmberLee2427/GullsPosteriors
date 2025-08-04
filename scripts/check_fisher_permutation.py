#!/usr/bin/env python
"""
check_fisher_permutation.py  --samples post_samples.npy --fisher fisher_cov.npy [--labels labels.txt]

Diagnostic:
1.  Reads posterior samples (shape = (nsamples, ndim)).
2.  Reads Fisher covariance matrix (ndim × ndim).
3.  Converts both to correlation matrices.
4.  Finds the permutation that maximises column similarity via the Hungarian algorithm.
5.  Reports the permutation and the mean absolute difference between the permuted Fisher correlation and the posterior correlation.
6.  If matplotlib is available, optionally saves a side-by-side heat-map figure.

Interpretation:
  • If mean |ΔR| becomes small after permutation, your Fisher columns were in the wrong order.
  • If it remains large, the Fisher approximation is simply poor.
"""
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compare Fisher covariance and posterior samples to check column order.")
parser.add_argument("--samples", required=True,
                    help="Path to numpy file containing posterior samples (shape: nsamples × ndim)")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--fisher", help="Path to numpy file containing Fisher covariance (ndim × ndim)")
group.add_argument("--lcfile", help="Path to .det.lc light-curve file; Fisher covariance will be computed from derivative columns")

parser.add_argument("--labels", help="Optional text file with one label per line (ndim lines)")
parser.add_argument("--plot", action="store_true", help="Save correlation heat-maps")
parser.add_argument("--outfile", default="fisher_vs_posterior.png", help="Filename for heat-maps when --plot is used")
parser.add_argument("--header_skip", type=int, default=12,
                    help="Number of header lines to skip before data rows in the .det.lc file (default: 12)")

args = parser.parse_args()

samples = np.load(args.samples)
if samples.ndim != 2:
    raise ValueError("samples array must be 2-D (nsamples, ndim)")
ns, ndim = samples.shape


# -----------------------------------------------------------------------------
# Obtain Fisher covariance
# -----------------------------------------------------------------------------

if args.fisher:
    fisher_cov = np.load(args.fisher)
else:
    # Compute covariance from derivative columns in .det.lc file
    import pandas as pd

    lcfile = args.lcfile

    # Determine actual number of columns by reading first non-comment line
    with open(lcfile, "r") as f:
        for _ in range(args.header_skip):
            next(f)
        # Now find the first non-comment line
        line = ""
        while line.startswith("#") or len(line.strip()) == 0:
            line = next(f)
        first_data_line = line
    ncols = len(first_data_line.split())

    # Build column names (borrowed from Data.load_data expected_columns list)
    expected_columns = [
        "Simulation_time", "measured_relative_flux", "measured_relative_flux_error",
        "true_relative_flux", "true_relative_flux_error", "observatory_code", "saturation_flag",
        "best_single_lens_fit", "parallax_shift_t", "parallax_shift_u", "BJD", "source_x", "source_y",
        "lens1_x", "lens1_y", "lens2_x", "lens2_y", "X", "Y", "Z"
    ]
    # Append generic dTheta names up to a generous limit
    expected_columns += [f"dTheta{i}" for i in range(1, 100)]
    names = expected_columns[:ncols]

    df = pd.read_csv(lcfile, sep=r"\s+", comment="#", names=names, header=None, skiprows=args.header_skip)

    if "measured_relative_flux_error" not in df.columns:
        raise ValueError("Column 'measured_relative_flux_error' not found in .det.lc file.")

    dtheta_cols = [c for c in df.columns if c.startswith("dTheta")]
    if len(dtheta_cols) == 0:
        raise ValueError("No dTheta* derivative columns found in .det.lc file — cannot compute Fisher matrix.")

    derivs_full = df[dtheta_cols].to_numpy()
    flux_err = df["measured_relative_flux_error"].to_numpy()
    inv_var = 1.0 / (flux_err ** 2)

    n_param_full = derivs_full.shape[1]

    # ---------------------------------------------------------------------
    # Some data sets append flux–parameter derivatives (F_S, F_B for each
    # band) after the main model parameters.  Those do *not* belong in the
    # Fisher matrix we want to compare with the posterior parameters.
    # If we detect extra columns, we conservatively drop the *last* ones.
    # ---------------------------------------------------------------------
    if n_param_full > ndim:
        drop = n_param_full - ndim
        print(f"Note: Found {n_param_full} dTheta columns but posterior ndim={ndim}.\n"
              f"      Assuming the last {drop} columns are flux-parameter derivatives and will be ignored.")
        derivs = derivs_full[:, :ndim]
    else:
        derivs = derivs_full

    n_param = derivs.shape[1]

    fisher_matrix = np.zeros((n_param, n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            val = np.sum(derivs[:, i] * inv_var * derivs[:, j])
            fisher_matrix[i, j] = val
            fisher_matrix[j, i] = val

    # Numerical stability: add tiny ridge if needed
    eps = 1e-12 * np.trace(fisher_matrix) / n_param
    fisher_matrix += np.eye(n_param) * eps

    fisher_cov = np.linalg.inv(fisher_matrix)

# Check dimension consistency
if fisher_cov.shape[0] != ndim:
    if fisher_cov.shape[0] > ndim:
        print(f"Warning: Fisher covariance has {fisher_cov.shape[0]} dimensions but posterior has {ndim}.\n"
              "         Truncating Fisher matrix to first ndim dimensions (assuming extra columns are flux parameters).")
        fisher_cov = fisher_cov[:ndim, :ndim]
    else:
        raise ValueError(f"Dimensionality mismatch: posterior ndim={ndim}, fisher ndim={fisher_cov.shape[0]}")


# Posterior covariance and correlation
Cpost = np.cov(samples, rowvar=False)
Dpost = np.sqrt(np.diag(Cpost))
Rpost = Cpost / np.outer(Dpost, Dpost)

# Fisher correlation
Df = np.sqrt(np.diag(fisher_cov))
Rf = fisher_cov / np.outer(Df, Df)

# Similarity matrix (absolute correlation of column vectors)
sim = np.abs(Rf) @ np.abs(Rpost)
rows, cols = linear_sum_assignment(-sim)  # maximise similarity
perm = cols

Rf_perm = Rf[np.ix_(perm, perm)]
score = np.mean(np.abs(Rf_perm - Rpost))

# Labels
if args.labels and Path(args.labels).exists():
    with open(args.labels) as f:
        labels = [line.strip() for line in f]
    if len(labels) != ndim:
        labels = [f"p{i}" for i in range(ndim)]
else:
    labels = [f"p{i}" for i in range(ndim)]

print("\n=== Fisher vs Posterior diagnostic ===")
print(f"ndim          : {ndim}")
print(f"Permutation    : {perm}")
print(f"mean |ΔR|      : {score:.3e}")
if score < 0.1:
    print("Result: Fisher columns likely mis-ordered. Apply permutation above.")
else:
    print("Result: Large discrepancy persists ⇒ Fisher approximation poor (or very different curvature).")

# Optional plot
if args.plot:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = ax[0].imshow(Rpost, vmin=-1, vmax=1, cmap='coolwarm')
    ax[0].set_title('Posterior corr')

    im1 = ax[1].imshow(Rf, vmin=-1, vmax=1, cmap='coolwarm')
    ax[1].set_title('Fisher corr (raw)')

    im2 = ax[2].imshow(Rf_perm, vmin=-1, vmax=1, cmap='coolwarm')
    ax[2].set_title('Fisher corr (perm)')

    for a in ax:
        a.set_xticks(range(ndim))
        a.set_yticks(range(ndim))
        a.set_xticklabels(labels, rotation=90, fontsize=6)
        a.set_yticklabels(labels, fontsize=6)

    # Add a single shared colour-bar to the right, clear of the images
    cbar = fig.colorbar(im0, ax=ax, location="right", shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')

    plt.savefig(args.outfile, dpi=300)
    print(f"Heatmaps saved to {args.outfile}") 