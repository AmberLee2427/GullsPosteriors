#unc_check_part2.py

# ### Fractional Errors

# package imports
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
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
MAKE_CORNER_PLOTS = False


def _parse_args_v2():
    args = sys.argv[1:]
    force_overwrite = False
    if args and args[0] == "f":
        force_overwrite = True
        args = args[1:]
    bins_to_run = args if args else COMPLETED_BINS
    for b in bins_to_run:
        if b not in COMPLETED_BINS:
            raise ValueError(f"Invalid mass bin specified: {b}. Valid options are: {COMPLETED_BINS}")
    return force_overwrite, bins_to_run


def _make_corner_plot_v2(samples, parameter_labels, truths_series, event_name, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    display_labels = [f"$\\log_{{10}}{p}$" if p in ["s", "q", "rho", "tE"] else f"${p}$" for p in parameter_labels]
    truths = []
    for p in parameter_labels:
        v = truths_series[p]
        if p in ["s", "q", "rho", "tE"]:
            truths.append(np.log10(v) if v > 0 else np.nan)
        else:
            truths.append(v)
    fig = corner.corner(samples, labels=display_labels, truths=np.array(truths), color="black")
    fig.savefig(output_path / f"{event_name}_comparison_corner.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_v2():
    force_overwrite, bins_to_run = _parse_args_v2()
    print(f"[part2] Starting with bins={bins_to_run}, force_overwrite={force_overwrite}", flush=True)
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
            if MAKE_CORNER_PLOTS and ((not plot_file.exists()) or force_overwrite):
                burn = event["samples"][len(event["samples"]) // 2:]
                _make_corner_plot_v2(burn, event["parameter_labels"], event["truths"], event_name, output_dir)

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
