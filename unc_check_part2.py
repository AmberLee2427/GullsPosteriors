#unc_check_part2.py

# ### Fractional Errors

# package imports
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner
import gc
from pathlib import Path
import pickle
import ot

rng = np.random.default_rng()

# deterministic v2 runner (single event-processing path)
COMPLETED_BINS = ["m-10", "m00", "m10", "m20", "m30", "m40"]
OT_THRESHOLD = 0.04
FRACTIONAL_THRESHOLD = 1 / 3
LOG_PARAMETERS = {"s", "q", "rho", "tE"}
FLUX_PARAMETER_PREFIXES = ("Fs_", "FB_", "Fbaseline_")


def _bool_from_env(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _is_flux_parameter(label):
    return label.startswith(FLUX_PARAMETER_PREFIXES)


def _display_label(label):
    if label in LOG_PARAMETERS:
        return f"$\\log_{{10}}{label}$"
    return f"${label}$"


def _truth_for_plot(label, truths_series):
    value = truths_series.get(label, np.nan)
    if label in LOG_PARAMETERS:
        return np.log10(value) if value > 0 else np.nan
    return value


def _summary_key(label):
    if label in LOG_PARAMETERS:
        return f"log_{label}_err"
    if label.startswith("Fs_"):
        return f"fs{label.split('_', 1)[1]}_err"
    if label.startswith("Fbaseline_"):
        return f"Fbase{label.split('_', 1)[1]}_err"
    if label.startswith("FB_"):
        return None
    return f"{label}_err"


def _mcmc_key(label):
    if _is_flux_parameter(label):
        return label
    return _summary_key(label)


def _fisher_cov_column(label):
    if label == "rho":
        return "dF_rs"
    if label.startswith("Fs_"):
        return f"dF_fs{label.split('_', 1)[1]}"
    if label.startswith("Fbaseline_"):
        return f"dF_Fbase{label.split('_', 1)[1]}"
    if label.startswith("FB_"):
        return None
    return f"dF_{label}"


def _parse_args_v2():
    args = sys.argv[1:]
    force_overwrite = _bool_from_env("UNC_FORCE_OVERWRITE", False)
    make_corner_plots = _bool_from_env("UNC_MAKE_CORNER_PLOTS", False)
    bins = []
    for arg in args:
        if arg in {"f", "--force-overwrite"}:
            force_overwrite = True
        elif arg == "--make-corner-plots":
            make_corner_plots = True
        elif arg == "--no-corner-plots":
            make_corner_plots = False
        else:
            bins.append(arg)
    bins_to_run = bins if bins else COMPLETED_BINS
    for b in bins_to_run:
        if b not in COMPLETED_BINS:
            raise ValueError(f"Invalid mass bin specified: {b}. Valid options are: {COMPLETED_BINS}")
    return force_overwrite, make_corner_plots, bins_to_run


def _add_fisher_ellipse(ax, center_x, center_y, cov_2x2):
    vals, vecs = np.linalg.eigh(cov_2x2)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * np.sqrt(2.3 * vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = mpatches.Ellipse(
        (center_x, center_y),
        width,
        height,
        angle=angle,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        zorder=10,
    )
    ax.add_patch(ellipse)


def _add_prior_ellipse(ax, center_x, center_y, sigma_x, sigma_y):
    cov_prior = np.diag([sigma_x**2, sigma_y**2])
    vals, vecs = np.linalg.eigh(cov_prior)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    for step in range(23):
        scale = step / 10.0
        width, height = 2 * np.sqrt(scale * vals)
        ellipse_filled = mpatches.Ellipse(
            (center_x, center_y),
            width,
            height,
            angle=angle,
            edgecolor="none",
            facecolor="red",
            alpha=0.005,
            zorder=1,
        )
        ax.add_patch(ellipse_filled)
    ellipse_edge = mpatches.Ellipse(
        (center_x, center_y),
        width,
        height,
        angle=angle,
        edgecolor="red",
        facecolor="none",
        linestyle=":",
        linewidth=1.0,
        alpha=0.3,
        zorder=2,
    )
    ax.add_patch(ellipse_edge)


def _make_corner_plot_v2(
    samples,
    parameter_labels,
    truths_series,
    fisher_unc,
    hdf5_unc,
    mcmc_unc,
    event_name,
    output_dir,
    fisher_cov=None,
    prior_unc=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ndim = len(parameter_labels)
    display_labels = [_display_label(label) for label in parameter_labels]
    plot_truths = np.array([_truth_for_plot(label, truths_series) for label in parameter_labels], dtype=float)

    fig = corner.corner(
        samples,
        labels=display_labels,
        truths=plot_truths,
        color="black",
        hist_kwargs={"color": "black"},
        plot_contours=True,
        plot_density=True,
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))

    if prior_unc is None:
        prior_unc = {}

    for i, label in enumerate(parameter_labels):
        diag_ax = axes[i, i]
        truth = plot_truths[i]
        summary_key = _summary_key(label)
        mcmc_key = _mcmc_key(label)

        if not np.isnan(truth):
            if summary_key and summary_key in fisher_unc.index and not np.isnan(fisher_unc[summary_key]):
                fisher_sigma = float(fisher_unc[summary_key])
                diag_ax.axvline(truth, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
                diag_ax.axvline(truth + fisher_sigma, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                diag_ax.axvline(truth - fisher_sigma, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)

            if summary_key and summary_key in prior_unc:
                prior_sigma = float(prior_unc[summary_key])
                if prior_sigma > 0:
                    diag_ax.axvline(truth + prior_sigma, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
                    diag_ax.axvline(truth - prior_sigma, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
                    xlim = diag_ax.get_xlim()
                    x_grid = np.linspace(xlim[0], xlim[1], 200)
                    gaussian = np.exp(-0.5 * ((x_grid - truth) / prior_sigma) ** 2) / (
                        prior_sigma * np.sqrt(2 * np.pi)
                    )
                    ylim = diag_ax.get_ylim()
                    if np.max(gaussian) > 0 and ylim[1] > 0:
                        gaussian_scaled = gaussian * (ylim[1] * 0.9 / np.max(gaussian))
                        diag_ax.plot(
                            x_grid,
                            gaussian_scaled,
                            color="red",
                            linestyle="-",
                            linewidth=0.5,
                            alpha=1.0,
                            zorder=5,
                        )

            if summary_key and summary_key in hdf5_unc.index and not np.isnan(hdf5_unc[summary_key]):
                hdf5_sigma = float(hdf5_unc[summary_key])
                diag_ax.axvline(truth + hdf5_sigma, color="green", linestyle="-.", linewidth=1.0, alpha=0.7)
                diag_ax.axvline(truth - hdf5_sigma, color="green", linestyle="-.", linewidth=1.0, alpha=0.7)

            if mcmc_key and mcmc_key in mcmc_unc:
                p50, err_minus, err_plus = mcmc_unc[mcmc_key]
                diag_ax.axvline(p50, color="black", linestyle="-", linewidth=1.5, alpha=0.7)
                diag_ax.axvline(p50 + err_plus, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                diag_ax.axvline(p50 - err_minus, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

        truth_text = ""
        if not np.isnan(truth):
            truth_text = f"{display_labels[i]} = {truth:.4f}"
            if summary_key and summary_key in fisher_unc.index and not np.isnan(fisher_unc[summary_key]):
                truth_text += f" $\\pm$ {float(fisher_unc[summary_key]):.4f}"
        if truth_text:
            diag_ax.text(0.5, 1.15, truth_text, color="blue", ha="center", va="center", transform=diag_ax.transAxes, fontsize=8)

        if mcmc_key and mcmc_key in mcmc_unc:
            p50, err_minus, err_plus = mcmc_unc[mcmc_key]
            post_text = f"{display_labels[i]} = ${p50:.4f}^{{+{err_plus:.4f}}}_{{-{err_minus:.4f}}}$"
            diag_ax.text(0.5, 1.05, post_text, color="black", ha="center", va="center", transform=diag_ax.transAxes, fontsize=8)

        for j in range(i):
            offdiag_ax = axes[i, j]
            truth_x = plot_truths[j]
            truth_y = plot_truths[i]
            if np.isnan(truth_x) or np.isnan(truth_y):
                continue

            label_j = parameter_labels[j]
            summary_key_j = _summary_key(label_j)

            if fisher_cov is not None:
                fisher_col_x = _fisher_cov_column(label_j)
                fisher_col_y = _fisher_cov_column(label)
                if fisher_col_x and fisher_col_y and fisher_col_x in fisher_cov.index and fisher_col_y in fisher_cov.index:
                    cov_fisher = np.array(
                        [
                            [fisher_cov.loc[fisher_col_x, fisher_col_x], fisher_cov.loc[fisher_col_x, fisher_col_y]],
                            [fisher_cov.loc[fisher_col_y, fisher_col_x], fisher_cov.loc[fisher_col_y, fisher_col_y]],
                        ],
                        dtype=float,
                    )
                    _add_fisher_ellipse(offdiag_ax, truth_x, truth_y, cov_fisher)
                    offdiag_ax.axvline(truth_x, color="blue", linestyle="-", linewidth=1.0, alpha=0.5)
                    offdiag_ax.axhline(truth_y, color="blue", linestyle="-", linewidth=1.0, alpha=0.5)

            if (
                summary_key
                and summary_key_j
                and summary_key in hdf5_unc.index
                and summary_key_j in hdf5_unc.index
                and not np.isnan(hdf5_unc[summary_key])
                and not np.isnan(hdf5_unc[summary_key_j])
            ):
                offdiag_ax.axvline(truth_x, color="green", linestyle="-.", linewidth=1.0, alpha=0.5)
                offdiag_ax.axhline(truth_y, color="green", linestyle="-.", linewidth=1.0, alpha=0.5)

            if (
                summary_key
                and summary_key_j
                and summary_key in prior_unc
                and summary_key_j in prior_unc
            ):
                offdiag_ax.axvline(truth_x, color="red", linestyle=":", linewidth=1.0, alpha=0.5)
                offdiag_ax.axhline(truth_y, color="red", linestyle=":", linewidth=1.0, alpha=0.5)
                sigma_x = float(prior_unc[summary_key_j])
                sigma_y = float(prior_unc[summary_key])
                if sigma_x > 0 and sigma_y > 0:
                    _add_prior_ellipse(offdiag_ax, truth_x, truth_y, sigma_x, sigma_y)

    if ndim >= 2:
        table_axes_indices = [(0, ndim - 2), (0, ndim - 1), (1, ndim - 2), (1, ndim - 1)]
        for row_idx, col_idx in table_axes_indices:
            axes[row_idx, col_idx].axis("off")
        table_ax = axes[0, -1]
    else:
        table_ax = axes[0, 0]

    table_data = [[
        "Param",
        r"Prior $\sigma$",
        r"Fisher $\sigma$",
        r"HDF5 $\sigma$",
        r"MCMC $\sigma_{-}$",
        r"MCMC $\sigma_{+}$",
    ]]
    for i, label in enumerate(parameter_labels):
        summary_key = _summary_key(label)
        mcmc_key = _mcmc_key(label)

        prior_val = float(prior_unc[summary_key]) if summary_key and summary_key in prior_unc else np.nan
        fisher_val = float(fisher_unc[summary_key]) if summary_key and summary_key in fisher_unc.index else np.nan
        hdf5_val = float(hdf5_unc[summary_key]) if summary_key and summary_key in hdf5_unc.index else np.nan

        prior_str = f"{prior_val:.4f}" if not np.isnan(prior_val) else "-"
        fisher_str = f"{fisher_val:.4f}" if not np.isnan(fisher_val) else "-"
        hdf5_str = f"{hdf5_val:.4f}" if not np.isnan(hdf5_val) else "-"

        if mcmc_key and mcmc_key in mcmc_unc:
            _, err_minus, err_plus = mcmc_unc[mcmc_key]
            mcmc_minus_str = f"{err_minus:.4f}"
            mcmc_plus_str = f"{err_plus:.4f}"
        else:
            mcmc_minus_str = "-"
            mcmc_plus_str = "-"

        table_data.append([display_labels[i], prior_str, fisher_str, hdf5_str, mcmc_minus_str, mcmc_plus_str])

    table = table_ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="upper right",
        colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(2.2, 2.5)

    for column_idx in range(6):
        cell = table[(0, column_idx)]
        cell.set_facecolor("#E0E0E0")
        cell.set_text_props(weight="bold", fontsize=9)

    for row_idx in range(1, len(table_data)):
        table[(row_idx, 1)].set_facecolor("#FFE6E6")
        table[(row_idx, 2)].set_facecolor("#E6F2FF")
        table[(row_idx, 3)].set_facecolor("#E6FFE6")
        table[(row_idx, 4)].set_facecolor("#F0F0F0")
        table[(row_idx, 5)].set_facecolor("#F0F0F0")

    fig.suptitle(event_name, y=1.0)
    fig.savefig(output_path / f"{event_name}_comparison_corner.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_v2():
    force_overwrite, make_corner_plots, bins_to_run = _parse_args_v2()
    print(
        f"[part2] Starting with bins={bins_to_run}, force_overwrite={force_overwrite}, "
        f"make_corner_plots={make_corner_plots}",
        flush=True,
    )
    data = {}
    for mass_bin in bins_to_run:
        print(f"[part2] Loading cache for {mass_bin}", flush=True)
        cache_path = Path.cwd() / f"data_cache_{mass_bin}.pkl"
        with cache_path.open("rb") as f:
            data[mass_bin] = pickle.load(f)

        event_list = data[mass_bin]["event_list"]
        data_dir = data[mass_bin]["data_dir"] + mass_bin + "/"
        output_dir = data_dir + "posteriors/"
        unphysical_rows = []
        print(f"[part2] Processing {mass_bin}: {len(event_list)} events", flush=True)

        for event_idx, (prm_file, event) in enumerate(event_list.items(), start=1):
            if event_idx == 1 or event_idx % 25 == 0:
                print(f"[part2] {mass_bin}: event {event_idx}/{len(event_list)} ({prm_file})", flush=True)
            mcmc_unc = event["mcmc_uncertainties"]
            fisher_unc_series = event["fisher_uncertainties"]
            parameter_file_labels = event["parameter_labels"]
            truths_series = event["truths"]

            log_truths = {}
            parameter_err_list = list(mcmc_unc.keys())
            for label in parameter_file_labels:
                if f"log_{label}_err" in parameter_err_list:
                    log_truths[f"log_{label}"] = np.log10(truths_series[label])
                elif f"{label}_err" in parameter_err_list:
                    log_truths[label] = truths_series[label]

            parameter_list = sorted(log_truths.keys())
            prior_sigmas = event["prior_uncertainties"]

            samples = event["samples"][:, :len(parameter_file_labels)]
            n_total = samples.shape[0]
            n = min(n_total // 2, 10000)
            start_idx = n_total - n

            param_index_lookup = {
                p: parameter_file_labels.index(p.replace("log_", ""))
                for p in parameter_list
            }

            prior_buffer = np.empty(n, dtype=np.float32)
            posterior_buffer = np.empty(n, dtype=np.float32)

            fractional_mcmc_error = {}
            mcmc_constrained = {}
            mcmc_prior_ot_distance = {}
            mcmc_prior_ot_constrained = {}
            fractional_fisher_error = {}
            fisher_constrained = {}
            comparison_metrics = {}

            for p in parameter_list:
                err_key = f"{p}_err"
                if err_key not in prior_sigmas:
                    continue

                p50, err_minus, err_plus = mcmc_unc[err_key]
                half_range = np.abs(err_plus + err_minus) / 2.0
                # denominator for fractional uncertainty
                base_label = p.replace("log_", "")
                if base_label == "alpha":
                    denom = 1.0  # compare absolute sigma for alpha
                elif base_label in {"piEN", "piEE"} and ("piEN" in truths_series.index and "piEE" in truths_series.index):
                    piE_mag = np.hypot(float(truths_series["piEN"]), float(truths_series["piEE"]))
                    denom = piE_mag if piE_mag > 0 else np.nan
                else:
                    denom = np.abs(log_truths[p])

                mcmc_frac = half_range / denom if (not np.isnan(denom) and denom != 0) else np.nan
                fractional_mcmc_error[err_key] = mcmc_frac
                mcmc_constrained[err_key] = (not np.isnan(mcmc_frac)) and (mcmc_frac <= FRACTIONAL_THRESHOLD)

                rng.standard_normal(size=n, dtype=np.float32, out=prior_buffer)
                prior_buffer *= float(prior_sigmas[err_key])
                prior_buffer += float(log_truths[p])
                prior_buffer.sort()
                posterior_buffer[:] = samples[start_idx:, param_index_lookup[p]]
                posterior_buffer.sort()
                distance = ot.lp.emd2_1d(prior_buffer, posterior_buffer)
                mcmc_prior_ot_distance[err_key] = float(np.sqrt(distance) / prior_sigmas[err_key])
                mcmc_prior_ot_constrained[err_key] = mcmc_prior_ot_distance[err_key] > OT_THRESHOLD

                fish = fisher_unc_series[err_key]
                fish_frac = np.abs(fish) / denom if (not np.isnan(denom) and denom != 0) else np.nan
                fractional_fisher_error[err_key] = fish_frac
                fisher_constrained[err_key] = (not np.isnan(fish_frac)) and (fish_frac <= FRACTIONAL_THRESHOLD)

                mcmc_sigma = half_range
                fisher_sigma = np.abs(float(fish))
                ratio = np.nan if fisher_sigma == 0 else mcmc_sigma / fisher_sigma
                mc_constrained_for_matrix = bool(mcmc_constrained[err_key] and mcmc_prior_ot_constrained[err_key])
                if mc_constrained_for_matrix and fisher_constrained[err_key]:
                    matrix_key = "A1" if (not np.isnan(ratio) and ratio >= 1.0) else "A2"
                elif mc_constrained_for_matrix and not fisher_constrained[err_key]:
                    matrix_key = "B"
                elif (not mc_constrained_for_matrix) and fisher_constrained[err_key]:
                    matrix_key = "C"
                else:
                    matrix_key = "D"

                comparison_metrics[err_key] = {
                    "mcmc_sigma": mcmc_sigma,
                    "fisher_sigma": fisher_sigma,
                    "ratio_mcmc_over_fisher": ratio,
                    "mcmc_fractional": mcmc_frac,
                    "fisher_fractional": fish_frac,
                    "mcmc_frac_constrained": mcmc_constrained[err_key],
                    "fisher_frac_constrained": fisher_constrained[err_key],
                    "ot_distance_scaled": mcmc_prior_ot_distance[err_key],
                    "ot_constrained": mcmc_prior_ot_constrained[err_key],
                    "mc_constrained_for_matrix": mc_constrained_for_matrix,
                    "matrix_key": matrix_key,
                }

            event["fractional_mcmc_error"] = fractional_mcmc_error
            event["mcmc_constrained_fractional"] = mcmc_constrained
            event["fractional_fisher_error"] = fractional_fisher_error
            event["fisher_constrained_fractional"] = fisher_constrained
            event["mcmc_prior_ot_distance"] = mcmc_prior_ot_distance
            event["mcmc_prior_ot_constrained"] = mcmc_prior_ot_constrained
            event["comparison_metrics"] = comparison_metrics

            has_unphysical_flux = False
            for label, tup in mcmc_unc.items():
                p50_flux = float(tup[0])
                if label.startswith("Fs_") and p50_flux < 0:
                    has_unphysical_flux = True
                    break
                if label.startswith("FB_") and p50_flux > 1:
                    has_unphysical_flux = True
                    break
            event["has_unphysical_flux"] = has_unphysical_flux
            if has_unphysical_flux:
                unphysical_rows.append(
                    {
                        "mass_bin": mass_bin,
                        "prm_file": prm_file,
                        "event_id": event.get("event_id"),
                        "field": event.get("field"),
                        "subrun": event.get("subrun"),
                    }
                )

            del prior_buffer, posterior_buffer
            gc.collect()

            event_name = f"{event['field']}_{event['subrun']}_{event['event_id']}"
            plot_file = Path(output_dir) / f"{event_name}_comparison_corner.png"
            if make_corner_plots and ((not plot_file.exists()) or force_overwrite):
                burn = event["samples"][len(event["samples"]) // 2:]
                _make_corner_plot_v2(
                    burn,
                    event["parameter_labels"],
                    event["truths"],
                    event.get("fisher_uncertainties", pd.Series(dtype=float)),
                    event.get("hdf5_uncertainties", pd.Series(dtype=float)),
                    event.get("mcmc_uncertainties", {}),
                    event_name,
                    output_dir,
                    fisher_cov=event.get("fisher_covariance"),
                    prior_unc=event.get("prior_uncertainties", {}),
                )

        minimal_cache = {
            "event_list": {
                key: {
                    "field": event["field"],
                    "subrun": event["subrun"],
                    "event_id": event["event_id"],
                    "parameter_labels": list(event["parameter_labels"]),
                    "truths": dict(event["truths"]),
                    "fisher_uncertainties": event.get("fisher_uncertainties", pd.Series(dtype=float)).to_dict(),
                    "hdf5_uncertainties": event.get("hdf5_uncertainties", pd.Series(dtype=float)).to_dict(),
                    "mcmc_uncertainties": dict(event.get("mcmc_uncertainties", {})),
                    "prior_uncertainties": dict(event.get("prior_uncertainties", {})),
                    "comparison_metrics": dict(event.get("comparison_metrics", {})),
                    "has_unphysical_flux": bool(event.get("has_unphysical_flux", False)),
                }
                for key, event in event_list.items()
            },
            "config": {
                "ot_threshold": OT_THRESHOLD,
                "fractional_threshold": FRACTIONAL_THRESHOLD,
            },
        }
        with open(Path.cwd() / f"minimal_cache_{mass_bin}.pkl", "wb") as f:
            pickle.dump(minimal_cache, f)

        unphys_path = Path.cwd() / f"unphysical_events_{mass_bin}.csv"
        pd.DataFrame(unphysical_rows).to_csv(unphys_path, index=False)
        print(
            f"[part2] Finished {mass_bin}: wrote minimal_cache_{mass_bin}.pkl and {unphys_path.name} "
            f"({len(unphysical_rows)} flagged)",
            flush=True,
        )


if __name__ == "__main__":
    _run_v2()
    raise SystemExit(0)
