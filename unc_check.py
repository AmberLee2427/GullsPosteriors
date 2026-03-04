
# # Fisher Matrix Uncertainty Consistency Checks
# 
# This script is hardcoded to work for a 6 filter gulls run and wzk obs group = 3.
# 
# Start a new Jupyter kernel using:
# ```bash
# python -m ipykernel install --user --name GullsPosteriors --display-name "Python (GullsPosteriors)"
# jupyter notebook --no-browser --port=8888
# ```

# ## Set Up


# package imports
import pandas as pd
import numpy as np
import sys
import yaml
import os
import matplotlib.pyplot as plt
import corner
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# data loading
completed_bins = ["m-10", "m00", "m10", "m20", "m30", "m40"]

args = sys.argv[1:]
force_reload = False
if args and args[0] == "f":
    force_reload = True
    args = args[1:]
bins_to_run = args if args else completed_bins

for bin in bins_to_run:
    if bin not in completed_bins:
        raise ValueError(f"Invalid mass bin specified: {bin}. Valid options are: {completed_bins}")

data = {}
for mass_bin in bins_to_run:
    data_dir = "../filter_selection/6f_overguide_" + mass_bin + "/"
    dstore = pd.HDFStore(data_dir + 'analysis/6f_overguide_' + mass_bin + '.det.hdf5','r')
    data[mass_bin] = {"data_dir": data_dir, "dstore": dstore, "wzk": None, "unc": None, "event_list": None}

    # Checking whats in the last hdf5 file
    print(dstore.info())

    # verifying the keys
    print(dstore.keys())

    # We need the wzk data, corresponding to `ObsGroup=3`.


    dstore = data[mass_bin]["dstore"]
    wzk = dstore['/overguide_6hcc_wzk']
    data[mass_bin]["wzk"] = wzk.copy()
    if isinstance(dstore, pd.HDFStore):
        dstore.close()
    data[mass_bin]["dstore"] = None  # free up memory

    # Show all columns when using head()
    pd.set_option('display.max_columns', None)
    # data["m00"]["wzk"].head()

    #print 5 rows to log
    print(data[mass_bin]["wzk"].head())
    print(data[mass_bin]["wzk"].shape)

    # Narrowing down the data to just what we need, so that it's a little less overwhelming and annoying to use.

    # Select event identifiers + parameter uncertainties
    wzk = data[mass_bin]["wzk"]
    keep = ['EventID','SubRun','Field']
    for col in wzk.columns:
        if col.startswith("ObsGroup_3") and col.endswith("_err"):
            keep.append(col)
        if "weight" in col:
            keep.append(col)
    data[mass_bin]["unc"] = wzk[keep].copy()
    data[mass_bin]["wzk"] = None  # free memory

    data[mass_bin]["unc"].head()




# ## Uncertainties from Each Source

import gc
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import pickle
from pathlib import Path


def build_event_catalog_for_mass_bin(mass_bin: str, *, force_reload: bool = False) -> dict:
    """
    load all event and lightcurve metadata, MCMC samples, and compute Fisher uncertainties 
    for a given mass bin. Results are cached in the `data` dictionary under the "event_list" 
    key for each mass bin, so that subsequent calls with `force_reload=False` will return 
    the cached results without recomputing. Set `force_reload=True` to refresh the cache 
    for a mass bin.
    
    :param mass_bin: Description
    :type mass_bin: str
    :param force_reload: Description
    :type force_reload: bool
    :return: Description
    :rtype: dict
    """
    
    start = time.perf_counter()
    print(f"Processing mass bin: {mass_bin}")
    entry = data[mass_bin]

    if not force_reload and entry.get("event_list"):
        print("  Cached event_list found; skipping rebuild. Use force_reload=True to refresh.")
        return entry["event_list"]

    unc = entry["unc"]
    data_dir = Path(entry["data_dir"]) / mass_bin
    output_dir = data_dir / "posteriors"

    event_list: dict[str, dict] = {}

    for filename in sorted(os.listdir(output_dir)):
        if "samples" not in filename:
            continue

        print(filename)
        # load the sampling parameter file
        prm_file = None
        for suffix in ("emcee_samples.npy", "dynesty_samples.npy", "post_samples.npy"):
            if suffix in filename:
                prm_file = filename.replace(suffix, "sampling.prm")
                break
        if prm_file is None:
            print(f"  Unable to infer prm filename for {filename}, skipping")
            continue

        if prm_file in event_list:
            print(f"  Skipping {prm_file} - already processed")
            continue

        prm_path = output_dir / prm_file
        if not prm_path.exists():
            print(f"  prm file {prm_file} not found, skipping")
            raise FileNotFoundError(f"prm file {prm_file} not found")
        with open(prm_path, "r") as f:
            prm_labels = yaml.safe_load(f)

        parameter_labels = prm_labels["model_config"]["parameter_labels"]
        event_id = int(prm_labels["event_truths"]["EventID"])
        field = int(prm_labels["event_truths"]["Field"])
        subrun = int(prm_labels["event_truths"]["SubRun"])
        lcname = prm_labels["event_truths"]["lcname"]
        truths = prm_labels["event_truths"]["params"]
        truths = truths[0 : len(parameter_labels)]
        truths_series = pd.Series(truths, index=parameter_labels)

        event_list[prm_file] = {
            "event_id": event_id,
            "field": field,
            "subrun": subrun,
            "lcname": lcname,
            "parameter_labels": parameter_labels,
            "truths": truths_series,
        }

        prior_sigmas = {}
        if prm_labels["prior_type"]["normal"] and not prm_labels["prior_type"]["unit_cube"]:
            print("  Prior type: Gaussian normal")
        if "prior_config" in prm_labels:
            for key, val in prm_labels["prior_config"].items():
                if key.startswith("sigma_"):
                    param_name = key.replace("sigma_", "")
                    if param_name.startswith("log"):
                        param_name = param_name.replace("log", "")
                        prior_sigmas[f"log_{param_name}_err"] = val
                    else:
                        prior_sigmas[f"{param_name}_err"] = val
        event_list[prm_file]["prior_uncertainties"] = prior_sigmas

        print(f"Processing EventID: {event_id}, Field: {field}, SubRun: {subrun}")
        column_rename = unc.columns.str.replace("ObsGroup_3_", "")
        unc.columns = column_rename
        event = unc.query("EventID == @event_id and Field == @field and SubRun == @subrun")
        if len(event) == 0:
            print(f"Event {event_id}_{field}_{subrun} not found in input data")
            continue
        event = event.iloc[0]
        event_list[prm_file]["simulation_metadata"] = event

        samples = np.load(output_dir / filename)
        print(samples.shape)

        blobs_filename = filename.replace("samples.npy", "blobs.npy")
        blobs_keys_filename = filename.replace("samples.npy", "blobs_keys.npy")
        blobs_path = output_dir / blobs_filename
        blobs_keys_path = output_dir / blobs_keys_filename

        if blobs_path.exists() and blobs_keys_path.exists():
            print(f"  Loading flux parameter blobs from {blobs_filename}")
            blobs = np.load(blobs_path)
            blobs_keys = np.load(blobs_keys_path, allow_pickle=True)
            blobs_keys = [k.decode() if isinstance(k, bytes) else str(k) for k in blobs_keys]
            print(f"  Blobs shape: {blobs.shape}")
            print(f"  Blobs columns: {list(blobs_keys)}")
            if blobs.shape[0] == samples.shape[0]:
                samples = np.hstack([samples, blobs])
                print(f"  Combined samples shape: {samples.shape}")
                parameter_labels = parameter_labels + list(blobs_keys)
            else:
                print(f"  Warning: Blobs shape mismatch ({blobs.shape[0]} vs {samples.shape[0]}), skipping")
        else:
            print("  No blobs files found, flux parameters not available")

        event_list[prm_file]["samples"] = samples
        event_list[prm_file]["parameter_labels"] = parameter_labels

        lc_file = data_dir / lcname
        header_lines = []
        with open(lc_file) as fh:
            for _ in range(16):
                header_lines.append(next(fh))

        fs_line = next(line for line in header_lines if line.startswith("#fs:"))
        fs_values = [float(x) for x in fs_line.split()[1:]]
        print(f"True source fluxes: {fs_values}")

        obs_group = prm_labels["plotting_config"]["obs_group"]
        obsgroup_line = next(line for line in header_lines if line.startswith(f"#Obsgroup: {obs_group}"))
        obs_group_band_indices = [int(x) for x in obsgroup_line.split()[-3:]]
        print(f"Observatory group {obs_group} uses band indices: {obs_group_band_indices}")

        for b in obs_group_band_indices:
            fs_key = f"Fs_{b}"
            fbaseline_key = f"Fbaseline_{b}"
            fb_key = f"FB_{b}"
            if fs_key not in parameter_labels:
                print(f"  Flux parameters for band {b} not in samples, skipping flux truths")
                continue
            print(f"  Adding flux truths for band {b}")
            Fs = fs_values[b]
            truths_series[f"Fs_{b}"] = Fs
            if fbaseline_key in parameter_labels:
                truths_series[f"Fbaseline_{b}"] = 1.0
            else:
                print(f"  Fbaseline parameter for band {b} not in samples, skipping")
            if fb_key in parameter_labels:
                truths_series[f"FB_{b}"] = 1.0 - Fs
            else:
                print(f"  FB parameter for band {b} not in samples, skipping")

        missing_truths = [label for label in parameter_labels if label not in truths_series.index]
        if missing_truths:
            print(f"Mass bin: {mass_bin}, EventID: {event_id}, Field: {field}, SubRun: {subrun}")
            raise ValueError(f"Missing truths for parameters: {missing_truths}")

        lc_data = np.loadtxt(lc_file, comments="#", skiprows=16)
        lc_columns = [
            "Simulation_time", "measured_relative_flux", "measured_relative_flux_error",
            "true_relative_flux", "true_relative_flux_error", "observatory_code",
            "saturation_flag", "best_single_lens_fit", "parallax_shift_t",
            "parallax_shift_u", "BJD", "source_x", "source_y", "lens1_x",
            "lens1_y", "lens2_x", "lens2_y", "parallax_shift_x",
            "parallax_shift_y", "parallax_shift_z", "dF_t0", "dF_tE",
            "dF_u0", "dF_alpha", "dF_s", "dF_q", "dF_rs", "dF_piEN",
            "dF_piEE", "dF_Fbase0", "dF_fs0",
            "dF_Fbase1", "dF_fs1", "dF_Fbase2", "dF_fs2",
            "dF_Fbase3", "dF_fs3", "dF_Fbase4", "dF_fs4",
            "dF_Fbase5", "dF_fs5",
        ]
        lc_df = pd.DataFrame(lc_data, columns=lc_columns)

        config_file = data_dir / ".gulls_config.json"
        config = json.load(open(config_file))
        observatories = config["obs_list"]
        lc_df = lc_df[lc_df["observatory_code"].isin(observatories)]
        print(f"Number of data points after filtering for observatories: {len(lc_df)}")

        dF_columns = [col for col in lc_df.columns if col.startswith("dF_")]
        for i in range(6):
            if i not in observatories and f"dF_Fbase{i}" in dF_columns:
                dF_columns.remove(f"dF_Fbase{i}")
            if i not in observatories and f"dF_fs{i}" in dF_columns:
                dF_columns.remove(f"dF_fs{i}")
        J = lc_df[dF_columns].to_numpy(dtype=float)
        sigma = lc_df["measured_relative_flux_error"].to_numpy(dtype=float)
        Jw = J / sigma[:, None]
        Fisher = Jw.T @ Jw
        try:
            Cov = np.linalg.inv(Fisher)
        except np.linalg.LinAlgError:
            print("Fisher matrix is singular, cannot invert")
            continue
        fisher_unc = np.sqrt(np.diag(Cov))
        Cov_df = pd.DataFrame(Cov, index=dF_columns, columns=dF_columns)
        dF_to_param = {}
        for dF_col in dF_columns:
            param = dF_col.replace("dF_", "")
            if param == "rs":
                param = "rho"
            if param in ["s", "q", "rho", "tE"]:
                dF_to_param[dF_col] = f"log_{param}_err"
            else:
                dF_to_param[dF_col] = f"{param}_err"
        fisher_unc_series = pd.Series(fisher_unc, index=dF_columns).rename(dF_to_param)

        print(f"Number of samples: {samples.shape[0]}")
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        mcmc_unc = {}
        print("Percentiles (16, 50, 84):")
        for i, label in enumerate(parameter_labels):
            p16, p50, p84 = percentiles[:, i]
            err_minus = p50 - p16
            err_plus = p84 - p50
            if label.startswith("Fs_") or label.startswith("FB_") or label.startswith("Fbaseline_"):
                mcmc_unc[label] = (p50, err_minus, err_plus)
                print(f"{label}: {p50:.3f} (+{err_plus:.3f}/-{err_minus:.3f})")
            else:
                label_key = f"log_{label}_err" if label in ["s", "q", "rho", "tE"] else f"{label}_err"
                mcmc_unc[label_key] = (p50, err_minus, err_plus)
                print(f"{label_key}: {p50:.3f} (+{err_plus:.3f}/-{err_minus:.3f})")
                if label_key in event.index:
                    print(f" hdf5 err: {event[label_key]:.3f}")
                if label_key in fisher_unc_series.index:
                    print(f" fisher err: {fisher_unc_series[label_key]:.3f}")

        event_list[prm_file]["mcmc_uncertainties"] = mcmc_unc
        event_list[prm_file]["fisher_uncertainties"] = fisher_unc_series
        event_list[prm_file]["fisher_covariance"] = Cov_df
        just_modelling_parameters = fisher_unc_series.index.tolist()
        event_list[prm_file]["hdf5_uncertainties"] = event[just_modelling_parameters]

        print("")

    entry["event_list"] = event_list
    duration = time.perf_counter() - start
    print(f"Finished {mass_bin} in {duration / 60:.1f} minutes")
    gc.collect()
    return event_list




for mass_bin in bins_to_run:
    build_event_catalog_for_mass_bin(mass_bin, force_reload=force_reload)
    entry = data[mass_bin]
    
    if isinstance(entry["dstore"], pd.HDFStore):
        entry["dstore"].close()
        entry["dstore"] = None    # or del entry["dstore"]

    cache_path = Path.cwd() / f"data_cache_{mass_bin}.pkl"  # adjust to the folder you prefer
    
    # Save
    with cache_path.open("wb") as f:
        pickle.dump(data[mass_bin], f)
