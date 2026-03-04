#!/usr/bin/env python3
"""Build final uncertainty comparison plots from cached mass-bin outputs.

Usage examples
--------------
python unc_check_part3_plots.py
python unc_check_part3_plots.py m-10 m00 m10 --exclude-unphysical-flux
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPLETED_BINS = ["m-10", "m00", "m10", "m20", "m30", "m40"]
PLOT_BINS = COMPLETED_BINS
CACHE_DIR = Path(".")
OUTDIR = Path("unc_check_outputs/part3_plots")
EXCLUDE_UNPHYSICAL_FLUX = True


def load_cached_bins(cache_dir: Path, bins: list[str]) -> dict[str, dict]:
    data = {}
    for mass_bin in bins:
        minimal_cache_path = cache_dir / f"minimal_cache_{mass_bin}.pkl"
        full_cache_path = cache_dir / f"data_cache_{mass_bin}.pkl"
        cache_path = minimal_cache_path if minimal_cache_path.exists() else full_cache_path
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cache file for {mass_bin}. Expected {minimal_cache_path.name} or {full_cache_path.name}")
        with cache_path.open("rb") as f:
            data[mass_bin] = pickle.load(f)
    return data


def has_unphysical_flux(event_data: dict) -> bool:
    if "has_unphysical_flux" in event_data:
        return bool(event_data["has_unphysical_flux"])
    mcmc_unc = event_data.get("mcmc_uncertainties", {})
    for label, tup in mcmc_unc.items():
        p50 = float(tup[0])
        if label.startswith("Fs_") and p50 < 0:
            return True
        if label.startswith("FB_") and p50 > 1:
            return True
    return False


def build_summary_arrays(data: dict[str, dict], exclude_unphysical_flux: bool):
    fisher_sigmas = []
    mcmc_sigmas = []
    param_names = []

    mcmc_fractional = []
    fisher_fractional = []

    param_data = defaultdict(lambda: {"fisher_frac": [], "mcmc_frac": [], "ratios": [], "bias": []})
    excluded = []

    for mass_bin, bin_data in data.items():
        event_list = bin_data["event_list"]
        for prm_file, event_data in event_list.items():
            if exclude_unphysical_flux and has_unphysical_flux(event_data):
                excluded.append(
                    {
                        "mass_bin": mass_bin,
                        "prm_file": prm_file,
                        "event_id": event_data.get("event_id"),
                        "field": event_data.get("field"),
                        "subrun": event_data.get("subrun"),
                    }
                )
                continue

            parameter_labels = event_data["parameter_labels"]
            fisher_unc = pd.Series(event_data.get("fisher_uncertainties", {}), dtype=float)
            mcmc_unc = event_data.get("mcmc_uncertainties", {})
            truths_series = pd.Series(event_data.get("truths", {}), dtype=float)

            for label in parameter_labels:
                label_key = f"log_{label}_err" if label in ["s", "q", "rho", "tE"] else f"{label}_err"

                if label_key not in fisher_unc.index or np.isnan(fisher_unc[label_key]):
                    continue

                fisher_sig = float(fisher_unc[label_key])
                if fisher_sig == 0:
                    continue

                if label_key in mcmc_unc:
                    p50, err_minus, err_plus = mcmc_unc[label_key]
                    mcmc_sig = (float(err_minus) + float(err_plus)) / 2.0
                    fisher_sigmas.append(fisher_sig)
                    mcmc_sigmas.append(mcmc_sig)
                    param_names.append(label_key.replace("_err", ""))

                truth_val = np.nan
                if label in ["s", "q", "rho", "tE"]:
                    if truths_series[label] > 0:
                        truth_val = np.log10(truths_series[label])
                else:
                    truth_val = truths_series[label]

                if not np.isnan(truth_val) and truth_val != 0:
                    fisher_fractional.append(fisher_sig / np.abs(truth_val))

                if label_key in mcmc_unc:
                    p50, err_minus, err_plus = mcmc_unc[label_key]
                    p50 = float(p50)
                    err_minus = float(err_minus)
                    err_plus = float(err_plus)

                    if p50 != 0 and not np.isnan(p50):
                        sigma_mcmc = np.sqrt(err_minus**2 + err_plus**2)
                        mcmc_frac = sigma_mcmc / np.abs(p50)
                        param_data[label_key]["mcmc_frac"].append(mcmc_frac)
                        mcmc_fractional.append(mcmc_frac)

                    if not np.isnan(truth_val):
                        param_data[label_key]["bias"].append(p50 - truth_val)

                    ratio = mcmc_sig / fisher_sig
                    param_data[label_key]["ratios"].append(ratio)

                    if not np.isnan(truth_val) and truth_val != 0:
                        param_data[label_key]["fisher_frac"].append(fisher_sig / np.abs(truth_val))

    return {
        "fisher_sigmas": np.array(fisher_sigmas),
        "mcmc_sigmas": np.array(mcmc_sigmas),
        "param_names": np.array(param_names),
        "mcmc_fractional": np.array(mcmc_fractional),
        "fisher_fractional": np.array(fisher_fractional),
        "param_data": param_data,
        "excluded": excluded,
    }


def build_matrix_rows(data: dict[str, dict], exclude_unphysical_flux: bool) -> pd.DataFrame:
    rows = []
    for mass_bin, bin_data in data.items():
        event_list = bin_data["event_list"]
        for prm_file, event_data in event_list.items():
            if exclude_unphysical_flux and has_unphysical_flux(event_data):
                continue
            event_name = f"{event_data.get('field')}_{event_data.get('subrun')}_{event_data.get('event_id')}"
            metrics = event_data.get("comparison_metrics", {})
            for param_err_key, m in metrics.items():
                rows.append(
                    {
                        "mass_bin": mass_bin,
                        "event_name": event_name,
                        "prm_file": prm_file,
                        "param_err_key": param_err_key,
                        "matrix_key": m.get("matrix_key"),
                        "ratio_mcmc_over_fisher": m.get("ratio_mcmc_over_fisher"),
                        "mcmc_fractional": m.get("mcmc_fractional"),
                        "fisher_fractional": m.get("fisher_fractional"),
                        "ot_distance_scaled": m.get("ot_distance_scaled"),
                        "ot_constrained": m.get("ot_constrained"),
                        "mc_constrained_for_matrix": m.get("mc_constrained_for_matrix"),
                    }
                )
    return pd.DataFrame(rows)


def save_poster_tables(data: dict[str, dict], summary: dict, outdir: Path, exclude_unphysical_flux: bool) -> None:
    matrix_df = build_matrix_rows(data, exclude_unphysical_flux=exclude_unphysical_flux)
    if matrix_df.empty:
        print("No comparison_metrics found in cache; skipping poster matrix tables.")
        return

    matrix_df.to_csv(outdir / "poster_matrix_rows.csv", index=False)

    matrix_counts = (
        matrix_df.groupby(["param_err_key", "matrix_key"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    matrix_counts.to_csv(outdir / "poster_matrix_counts_by_parameter.csv", index=False)

    ratio_df = matrix_df[np.isfinite(matrix_df["ratio_mcmc_over_fisher"])].copy()
    if not ratio_df.empty:
        constrained = ratio_df[ratio_df["matrix_key"].isin(["A1", "A2"])].copy()
        param_summary = (
            constrained.groupby("param_err_key", as_index=False)["ratio_mcmc_over_fisher"]
            .agg(["count", "median", "mean", "std"])
            .reset_index()
            .rename(columns={"median": "median_ratio", "mean": "mean_ratio", "std": "std_ratio"})
        )
        param_summary["median_percent_diff"] = 100.0 * (param_summary["median_ratio"] - 1.0)
        param_summary = param_summary.sort_values("count", ascending=False)
        param_summary.to_csv(outdir / "poster_parameter_ratio_summary_constrained.csv", index=False)

        with (outdir / "poster_key_messages.txt").open("w") as f:
            f.write("Preliminary poster messages (constrained subset: A1 + A2)\n")
            f.write("=======================================================\n")
            for _, row in param_summary.iterrows():
                f.write(
                    f"{row['param_err_key']}: Fisher is {row['median_percent_diff']:+.1f}% relative to MCMC median (n={int(row['count'])})\n"
                )

    if summary["excluded"]:
        excluded_df = pd.DataFrame(summary["excluded"])
        excluded_df.to_csv(outdir / "poster_excluded_unphysical_flux_events.csv", index=False)


def save_global_plots(summary: dict, data: dict, outdir: Path) -> None:
    fisher_sigmas = summary["fisher_sigmas"]
    mcmc_sigmas = summary["mcmc_sigmas"]
    param_names = summary["param_names"]
    mcmc_fractional = summary["mcmc_fractional"]
    fisher_fractional = summary["fisher_fractional"]

    if len(fisher_sigmas) == 0 or len(mcmc_sigmas) == 0:
        print("No matched Fisher/MCMC entries found; skipping global plot.")
        return

    ratios = mcmc_sigmas / fisher_sigmas
    matrix_df = build_matrix_rows(data, exclude_unphysical_flux=EXCLUDE_UNPHYSICAL_FLUX)
    constrained_mask = matrix_df['matrix_key'].isin(['A1', 'A2']) & np.isfinite(matrix_df['ratio_mcmc_over_fisher'])
    constrained_ratios = matrix_df.loc[constrained_mask, 'ratio_mcmc_over_fisher'].values
    # Log-transform ratios for readability
    log_ratios = np.log10(ratios[np.isfinite(ratios)])
    log_constrained_ratios = np.log10(constrained_ratios[np.isfinite(constrained_ratios)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Ratio Distribution (log scale)
    ax = axes[0]
    bins = np.linspace(min(log_ratios), max(log_ratios), 30)
    hist_all, _ = np.histogram(log_ratios, bins=bins)
    hist_constrained, _ = np.histogram(log_constrained_ratios, bins=bins)
    ax.bar(bins[:-1], hist_all / hist_all.sum(), width=np.diff(bins), color="black", alpha=0.7, edgecolor="black", label="All")
    ax.step(bins[:-1], hist_constrained / hist_constrained.sum(), where="mid", color="red", linewidth=2, label="Constrained", fillstyle='none')
    ax.axvline(0, color="r", linestyle="--", linewidth=2, label="MCMC = Fisher (log10=0)")
    ax.axvline(np.median(log_ratios), color="blue", linestyle="-", linewidth=2, label=f"Median={np.median(log_ratios):.2f}")
    ax.set_xlabel("log10(MCMC $\\sigma$ / Fisher $\\sigma$)")
    ax.set_ylabel("Normalized Count")
    ax.set_title("Ratio Distribution (log scale)")
    ax.legend()

    # Panel 2: Cumulative Fractional Uncertainties (robust CDF, log x-axis)
    ax = axes[1]
    # All MCMC
    mcmc_finite = mcmc_fractional[np.isfinite(mcmc_fractional) & (mcmc_fractional > 0)]
    if len(mcmc_finite) > 0:
        sorted_mcmc = np.sort(mcmc_finite)
        cumulative_mcmc = np.arange(1, len(sorted_mcmc) + 1) / len(sorted_mcmc)
        ax.plot(sorted_mcmc, cumulative_mcmc, color="black", linewidth=2, label="MCMC (all)")
    # MCMC (constrained): A1, A2, B
    mcmc_constrained_mask = matrix_df['matrix_key'].isin(['A1', 'A2', 'B']) & np.isfinite(matrix_df['mcmc_fractional']) & (matrix_df['mcmc_fractional'] > 0)
    mcmc_constrained = matrix_df.loc[mcmc_constrained_mask, 'mcmc_fractional'].values
    if len(mcmc_constrained) > 0:
        sorted_constrained = np.sort(mcmc_constrained)
        cumulative_constrained = np.arange(1, len(sorted_constrained) + 1) / len(sorted_constrained)
        ax.plot(sorted_constrained, cumulative_constrained, color="red", linewidth=2, label="MCMC (constrained)", linestyle="--")
    # Fisher (all)
    fisher_finite = fisher_fractional[np.isfinite(fisher_fractional) & (fisher_fractional > 0)]
    if len(fisher_finite) > 0:
        sorted_fisher = np.sort(fisher_finite)
        cumulative_fisher = np.arange(1, len(sorted_fisher) + 1) / len(sorted_fisher)
        ax.plot(sorted_fisher, cumulative_fisher, color="blue", linewidth=2, label="Fisher (all)")
    # Fisher (constrained): A1, A2, C
    fisher_constrained_mask = matrix_df['matrix_key'].isin(['A1', 'A2', 'C']) & np.isfinite(matrix_df['fisher_fractional']) & (matrix_df['fisher_fractional'] > 0)
    fisher_constrained = matrix_df.loc[fisher_constrained_mask, 'fisher_fractional'].values
    if len(fisher_constrained) > 0:
        sorted_fisher_constrained = np.sort(fisher_constrained)
        cumulative_fisher_constrained = np.arange(1, len(sorted_fisher_constrained) + 1) / len(sorted_fisher_constrained)
        ax.plot(sorted_fisher_constrained, cumulative_fisher_constrained, color="magenta", linewidth=2, label="Fisher (constrained)", linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Fractional Uncertainty ($\\sigma$ / |value|)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative Fractional Uncertainties")
    ax.set_xscale('log')
    ax.legend()

    # Panel 3: Parameter-wise Median Ratios (bar graph, overlay constrained median as black outline, show n above bars)
    ax = axes[2]
    unique_params = sorted(set(param_names.tolist()))
    param_median_ratios = []
    param_constrained_medians = []
    param_constrained_counts = []
    for param in unique_params:
        mask = param_names == param
        param_median_ratios.append(np.median(ratios[mask]))
        # Constrained overlay: median only
        param_constrained = matrix_df[(matrix_df['param_err_key'] == param + '_err') & constrained_mask]['ratio_mcmc_over_fisher'].values
        if len(param_constrained) > 0:
            param_constrained_medians.append(np.median(param_constrained))
            param_constrained_counts.append(len(param_constrained))
        else:
            param_constrained_medians.append(np.nan)
            param_constrained_counts.append(0)
    x_pos = np.arange(len(unique_params))
    colors = ["green" if r < 1.0 else "orange" if abs(r - 1.0) < 0.2 else "red" for r in param_median_ratios]
    # Main bar graph
    ax.bar(x_pos, param_median_ratios, color=colors, alpha=0.7, edgecolor="black", label="All")
    # Overlay constrained median as black outline bar (no fill)
    for i, median_val in enumerate(param_constrained_medians):
        if not np.isnan(median_val):
            ax.bar(x_pos[i], median_val, color='none', edgecolor='black', linewidth=2, width=0.8, label="Constrained" if i == 0 else "")
            # Add count above bar
            ax.text(x_pos[i], median_val, f"n={param_constrained_counts[i]}", ha='center', va='bottom', fontsize=9, color='black', fontweight='bold', rotation=0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(unique_params, rotation=45, ha="right")
    ax.set_ylabel("Median MCMC $\\sigma$ / Fisher $\\sigma$")
    ax.set_title("Parameter-wise Median Ratios")
    ax.legend()

    plt.tight_layout()
    plt.savefig(outdir / "global_uncertainty_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_param_diagnostics(summary: dict, data: dict, outdir: Path) -> None:
    param_data = summary["param_data"]

    matrix_df = build_matrix_rows(data, exclude_unphysical_flux=EXCLUDE_UNPHYSICAL_FLUX)

    for param_name, data in param_data.items():
        if len(data["mcmc_frac"]) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        fig.suptitle(f"Parameter: {param_name}")

        # Panel 1: CDFs (log10 x-axis)
        ax = axes[0]
        # MCMC
        if len(data["mcmc_frac"]) > 0:
            mcmc_arr_raw = np.array(data["mcmc_frac"])
            mcmc_arr = np.sort(np.log10(mcmc_arr_raw[mcmc_arr_raw > 0]))
            cdf = np.arange(1, len(mcmc_arr) + 1) / len(mcmc_arr)
            ax.plot(mcmc_arr, cdf, color="black", linewidth=2, label="MCMC")
        # Fisher
        if len(data["fisher_frac"]) > 0:
            fisher_arr_raw = np.array(data["fisher_frac"])
            fisher_arr = np.sort(np.log10(fisher_arr_raw[fisher_arr_raw > 0]))
            cdf = np.arange(1, len(fisher_arr) + 1) / len(fisher_arr)
            ax.plot(fisher_arr, cdf, color="blue", linewidth=2, label="Fisher")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Cumulative Fractional Uncertainties (log scale)")
        ax.set_xlabel("log10(Fractional Uncertainty)")
        ax.legend()

        # Panel 2: Uncertainty Ratio Distribution (log10 x-axis)
        ax = axes[1]
        if len(data["ratios"]) > 0:
            ratios_arr = np.array(data["ratios"])
            log_ratios = np.log10(ratios_arr[ratios_arr > 0])
            # Normalize histogram in log10 space
            hist_all, bins = np.histogram(log_ratios, bins=20)
            print(f"[Panel 2] {param_name} log10(ratio) bins: {bins}")
            ax.bar(bins[:-1], hist_all / hist_all.sum(), width=np.diff(bins), color="black", alpha=0.7, edgecolor="black", label="All")
            # Overlay constrained samples
            constrained = matrix_df[(matrix_df['param_err_key'] == param_name) & matrix_df['matrix_key'].isin(['A1', 'A2'])]['ratio_mcmc_over_fisher'].values
            constrained = constrained[constrained > 0]
            if len(constrained) > 0:
                log_constrained = np.log10(constrained)
                hist_constrained, _ = np.histogram(log_constrained, bins=bins)
                ax.step(bins[:-1], hist_constrained / hist_constrained.sum(), where="mid", color="red", linewidth=2, label="Constrained", fillstyle='none')
            #ax.axvline(0.0, color="r", linestyle="--", linewidth=2, label="MCMC = Fisher (log10=0)")
            ax.axvline(np.median(log_ratios), color="blue", linestyle="-", linewidth=2, label=f"Median = {np.median(log_ratios):.2f}")
            ax.set_xlabel("log10(MCMC σ / Fisher σ)")
            ax.set_xlim(bins[0], bins[-1])
        ax.set_title("Uncertainty Ratio Distribution (log scale)")
        ax.legend(loc='upper right')

        # Panel 3: Posterior Bias Distribution
        ax = axes[2]
        if len(data["bias"]) > 0:
            bias_arr = np.array(data["bias"])
            hist_bias, bins_bias = np.histogram(bias_arr, bins=20)
            print(f"[Panel 3] {param_name} bias bins: {bins_bias}")
            ax.hist(bias_arr, bins=bins_bias, color="black", alpha=0.7, edgecolor="black")
            # Overlay outline-only constrained samples
            constrained_bias = matrix_df[(matrix_df['param_err_key'] == param_name) & matrix_df['matrix_key'].isin(['A1', 'A2'])]['ratio_mcmc_over_fisher'].values
            if len(constrained_bias) > 0:
                ax.hist(constrained_bias, bins=bins_bias, histtype="step", color="black", linewidth=2, alpha=1.0, zorder=10)
            ax.axvline(0.0, color="r", linestyle="--", linewidth=2)
            ax.axvline(np.median(bias_arr), color="blue", linestyle="-", linewidth=2)
        ax.set_title("Posterior Bias Distribution")

        plt.tight_layout()
        plt.savefig(outdir / f"param_diagnostics_{param_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    invalid = sorted(set(PLOT_BINS) - set(COMPLETED_BINS))
    if invalid:
        raise ValueError(f"Invalid mass bins: {invalid}. Valid options: {COMPLETED_BINS}")

    cache_dir = CACHE_DIR.resolve()
    outdir = OUTDIR.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_cached_bins(cache_dir, PLOT_BINS)
    summary = build_summary_arrays(data, exclude_unphysical_flux=EXCLUDE_UNPHYSICAL_FLUX)

    if summary["excluded"]:
        excluded_df = pd.DataFrame(summary["excluded"])
        excluded_df.to_csv(outdir / "excluded_unphysical_flux_events.csv", index=False)
        print(f"Excluded {len(excluded_df)} events with unphysical flux posteriors.")

    save_global_plots(summary, data, outdir)
    save_param_diagnostics(summary, data, outdir)
    save_poster_tables(data, summary, outdir, exclude_unphysical_flux=EXCLUDE_UNPHYSICAL_FLUX)
    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
