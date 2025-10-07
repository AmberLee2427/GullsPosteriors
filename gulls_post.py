#!/usr/bin/env python3
"""Refactored CLI for posterior sampling on gull events.

- Uses argparse instead of manual sys.argv parsing
- Organizes logic into small functions for readability and testability
- Keeps behavior compatible with gulls_post_emcee_adaptive_BI.py

Usage (compatible):
  python gulls_post.py NEVENTS DATA_PATH [options]
"""
import os
import sys
import time
import warnings
import argparse
import numpy as np
import pickle
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    warnings.warn('matplotlib not available; plotting disabled for this session')
import yaml

from Data import Data
from Parallax import Parallax
from Event import Event
from Fit import Fit
from Orbit import Orbit
try:
    from VBMicrolensing import VBMicrolensing
except ImportError:
    print("Warning: VBMicrolensing not available. Some functionality may be limited.")
    VBMicrolensing = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run Bayesian posterior sampling on gravitational microlensing events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic EMCEE run on 5 events with default settings
  python gulls_post.py 5 Fisher_overguide_m40/
  
  # High-resolution run with frequent checkpoints and all plots
  python gulls_post.py 3 FishVBM_Therr_m20/ -s emcee -t 8 -n 5000 -nstep 25 -f ictpf
  
  # Dynesty nested sampling with Fisher-informed priors
  python gulls_post.py 1 Fisher_overguide_m40/ -s dynesty -fp -prior normal-unit-cube
  
  # Fast test run with minimal plots
  python gulls_post.py 1 Fisher_overguide_m40/ -n 100 -nstep 10 -f ic

Plot flags (-f):
  i = initial diagnostic plots (lightcurve, caustic)
  c = chain/trace plots during sampling  
  t = trace plots (dynesty)
  p = posterior corner plots
  f = final model overlay plots
  n = disable all plots
  
Samplers:
  emcee   = Ensemble MCMC sampler (default)
  dynesty = Nested sampling
  
Prior types:
  normal           = Normal priors in physical space (default for emcee)
  uniform          = Uniform priors in physical space  
  normal-unit-cube = Normal priors in unit cube (default for dynesty)
  uniform-unit-cube= Uniform priors in unit cube
        """)

    # Required arguments
    p.add_argument("nevents", type=int, nargs="?", default=None,
                   help="Number of events to process (required unless --events-file is used)")
    p.add_argument("path", 
                   help="Directory containing data challenge files (.lc, .hdf5)")

    event_group = p.add_argument_group("Event Selection")
    event_group.add_argument("--events-file", dest="events_file", default=None,
                             help="Path to a text file listing events to process (one per line)")
    event_group.add_argument("--obs-group", dest="obs_group", type=int, default=None,
                             help="Index into OBS_GROUPS from .prm to select observatory subset (0-indexed)")

    # Sampling configuration
    sampling_group = p.add_argument_group("Sampling Configuration")
    sampling_group.add_argument("-s", dest="sampler", choices=["emcee", "dynesty"], 
                               default="emcee", help="Sampling algorithm (default: emcee)")
    sampling_group.add_argument("-t", dest="threads", type=int, default=1, 
                               help="Number of parallel threads for emcee (default: 1)")
    sampling_group.add_argument("-n", dest="n_samples", type=int, default=1000,
                               help="Number of posterior samples to collect (default: 1000)")
    sampling_group.add_argument("-nstep", dest="n_step", type=int, default=100,
                               help="Steps between checkpoints/plots (default: 100)")

    # Burn-in configuration  
    burnin_group = p.add_argument_group("Burn-in Configuration")
    burnin_group.add_argument("-adapt", dest="adaptive_burnin", action="store_true",
                             help="Enable adaptive burn-in with prior expansion")
    burnin_group.add_argument("-nbimin", dest="burnin_min_steps", type=int, default=500,
                             help="Minimum burn-in steps (default: 500)")
    burnin_group.add_argument("-nbimax", dest="burnin_max_steps", type=int, default=1000,
                             help="Maximum burn-in steps (default: 1000)")
    burnin_group.add_argument("-nbistep", dest="burnin_stepi", type=int, default=200,
                             help="Steps between burn-in checkpoints (default: 200)")

    # Model configuration
    model_group = p.add_argument_group("Model Configuration") 
    model_group.add_argument("-noLOM", dest="no_lom", action="store_true",
                            help="Disable lens orbital motion (LOM) parameters")
    model_group.add_argument("-prior", dest="prior", 
                            choices=["normal", "uniform", "uniform-unit-cube", "normal-unit-cube"],
                            help="Prior distribution type (auto-selected by sampler if not specified)")
    model_group.add_argument("-fp", dest="use_fisher_prior", action="store_true",
                            help="Use Fisher matrix uncertainties to inform prior widths")

    # Output configuration
    output_group = p.add_argument_group("Output Configuration")
    output_group.add_argument("-f", dest="plots", default="ictpf",
                             help="Plot flags: i,c,t,p,f or n for none (default: ictpf)")
    output_group.add_argument("-sort", dest="sort", default="alphanumeric",
                             help="Event sorting method (default: alphanumeric)")

    return p.parse_args(argv)


def load_event_list(file_path):
    resolved_path = os.path.expanduser(os.path.expandvars(file_path))
    if not os.path.isfile(resolved_path):
        sys.exit(f"Event list file '{file_path}' does not exist.")

    events = []
    with open(resolved_path, 'r') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.lstrip().startswith('#'):
                continue

            tokens = stripped.split()
            normalized_entry = None
            if len(tokens) >= 3:
                try:
                    event_id = int(float(tokens[0]))
                    sub_run = int(float(tokens[1]))
                    field = int(float(tokens[2]))
                    normalized_entry = (event_id, sub_run, field)
                except ValueError:
                    normalized_entry = None

            if normalized_entry is not None:
                events.append(normalized_entry)
            else:
                events.append(stripped)

    if not events:
        sys.exit(f"Event list file '{file_path}' did not contain any usable entries.")

    return events


def derive_plot_flags(args):
    # Defaults by sampler
    plot_chain = args.sampler == "emcee"
    plot_trace = args.sampler == "dynesty"
    plot_post = args.sampler == "emcee"
    plot_run = args.sampler == "dynesty"
    plot_initial = True
    plot_final = True

    flags = args.plots or ""
    if "n" in flags:
        return False, False, False, False, False, False

    plot_initial = "i" in flags
    plot_chain = "c" in flags
    plot_trace = "t" in flags
    plot_post = "p" in flags
    if "f" in flags:
        plot_final = True
    else:
        plot_final = False

    # Ensure minimums for each sampler when final plots requested
    if plot_final:
        if args.sampler == "dynesty":
            plot_trace = True
            plot_run = True
        if args.sampler == "emcee":
            plot_post = True

    return plot_initial, plot_chain, plot_post, plot_trace, plot_run, plot_final


def compute_prior_setup(LOM_enabled):
    if LOM_enabled:
        ndim = 12
        labels = ["s","q","rho","u0","alpha","t0","tE","piEE","piEN","i","phase","period"]
        p_unc = np.array([0.05,0.1,0.5,0.1,0.05,0.5,2.5,5.0,5.0, np.pi/2.0, np.pi/2.0, 0.2])
        p_unc_log = np.array([0.1, 0.5, 0.5, 0.1, 0.05])
        prange_log = p_unc_log * 2.0
        lin_idx = [3,4,5,7,8,9,10]
        prange_linear = p_unc[lin_idx] * 2.0
    else:
        ndim = 9
        labels = ["s","q","rho","u0","alpha","t0","tE","piEE","piEN"]
        p_unc = np.array([0.05,0.1,0.5,0.1,0.05,0.5,2.5,5.0,5.0])
        p_unc_log = np.array([0.1, 0.5, 0.5, 0.1])
        prange_log = p_unc_log * 2.0
        lin_idx = [3,4,5,7,8]
        prange_linear = p_unc[lin_idx] * 2.0
    return ndim, labels, p_unc, prange_log, prange_linear


def choose_prior_type(args):
    if args.prior:
        return args.prior
    if args.sampler == "dynesty":
        return "normal-unit-cube"
    return "normal"


def save_run_parameters(args, event_name, path, truths, ndim, labels, 
                        prange_linear, prange_log, prior_type, start_time, fit_obj, gamma):
    """Save sampling run parameters to a .prm file in YAML format.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    event_name : str
        Name of the current event
    path : str
        Output directory path
    truths : dict
        Event truth parameters
    ndim : int
        Number of model dimensions
    labels : list
        Parameter labels
    prange_linear : array_like
        Linear prior ranges
    prange_log : array_like
        Log prior ranges  
    prior_type : str
        Type of prior used
    fit_obj : Fit object
        Fit object containing sigma parameters for priors and prior type.
    gamma : float
        Gamma parameter from data object
    """ 
    # Build parameter dictionary
    run_params = {
        'run_info': {
            'event_name': event_name,
            'start_time': time.ctime(start_time),
            'start_timestamp': start_time,
            'command_line': ' '.join(sys.argv),
            'working_directory': os.getcwd(),
        },
        'sampling_config': {
            'sampler': args.sampler,
            'threads': args.threads,
            'n_samples': args.n_samples,
            'n_step': args.n_step,
            'adaptive_burnin': args.adaptive_burnin,
            'burnin_min_steps': args.burnin_min_steps,
            'burnin_max_steps': args.burnin_max_steps,
            'burnin_stepi': args.burnin_stepi,
            'prior_type': prior_type,
            'use_fisher_prior': args.use_fisher_prior,
            'LOM_enabled': not args.no_lom,
        },
        'model_config': {
            'ndim': ndim,
            'parameter_labels': labels,
            'prange_linear': prange_linear.tolist() if hasattr(prange_linear, 'tolist') else list(prange_linear),
            'prange_log': prange_log.tolist() if hasattr(prange_log, 'tolist') else list(prange_log),
        },
        'prior_type': {
            'normal': fit_obj.normal,
            'unit_cube': fit_obj.unit_cube
        },
        'prior_config': {
            # Save sigma parameters for normal priors
            'sigma_fb': fit_obj.sigma_fb,
            'sigma_logs': fit_obj.sigma_logs if not fit_obj.unit_cube else None,
            'sigma_logq': fit_obj.sigma_logq if not fit_obj.unit_cube else None,
            'sigma_logrho': fit_obj.sigma_logrho if not fit_obj.unit_cube else None,
            'sigma_u0': fit_obj.sigma_u0 if not fit_obj.unit_cube else None,
            'sigma_alpha': fit_obj.sigma_alpha if not fit_obj.unit_cube else None,
            'sigma_t0': fit_obj.sigma_t0 if fit_obj.normal and not fit_obj.unit_cube else None,
            'sigma_logtE': fit_obj.sigma_logtE if fit_obj.normal and not fit_obj.unit_cube else None,
            'sigma_piEE': fit_obj.sigma_piEE if fit_obj.normal and not fit_obj.unit_cube else None,
            'sigma_piEN': fit_obj.sigma_piEN if fit_obj.normal and not fit_obj.unit_cube else None
        },
        'plotting_config': {
            'plot_flags': args.plots,
            'sort_method': args.sort,
            'obs_group': args.obs_group,
        },
        'event_truths': {
            # Save key truth parameters (avoid massive arrays)
            'EventID': truths.get('EventID'),
            'Field': truths.get('Field'),
            'SubRun': truths.get('SubRun'),
            'lcname': truths.get('lcname'),
            'gamma': truths.get('gamma', gamma),
            'params': truths['params'].tolist() if hasattr(truths.get('params'), 'tolist') else truths.get('params'),
        }
    }
    
    # Save to .prm file
    prm_filename = path + f"posteriors/{event_name}_sampling.prm"
    try:
        with open(prm_filename, 'w') as f:
            yaml.dump(run_params, f, default_flow_style=False, indent=2)
        print(f"Saved sampling parameters to {prm_filename}")
    except Exception as e:
        raise Exception("YAML saving failed") from e


def build_plot_titles(LOM_enabled):
    if LOM_enabled:
        ts = ("s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, "
              "\ntE=%.2f, piEE=%.2f, piEN=%.2f, i=%.2f, phase=%.2f, period=%.2f")
    else:
        ts = ("s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, "
              "\ntE=%.2f, piEE=%.2f, piEN=%.2f")
    return ts


def run(args):
    start_time = time.time()
    print("Start time =", start_time)

    path = args.path
    if not path.endswith("/"):
        path += "/"

    if args.events_file:
        event_identifiers = load_event_list(args.events_file)
        total_events = len(event_identifiers)
        args.nevents = total_events
        print(f"Loaded {total_events} event(s) from {args.events_file}.")
    else:
        if args.nevents is None:
            sys.exit("Must specify NEVENTS when --events-file is not provided.")
        total_events = args.nevents
        event_identifiers = None

    LOM_enabled = not args.no_lom
    print("Lens Orbit Motion (LOM) is {}.".format("ENABLED" if LOM_enabled else "DISABLED"))
    print("{} Fisher uncertainties to inform prior ranges.".format(
        "Using" if args.use_fisher_prior else "NOT using"))

    ndim, labels, p_unc, prange_log, prange_linear = compute_prior_setup(LOM_enabled)

    adaptive_burnin = args.adaptive_burnin
    if adaptive_burnin and args.use_fisher_prior:
        sys.exit("Fisher informed prior ranges not supported with adaptive burn-in.")

    prior_type = choose_prior_type(args)

    plot_initial, plot_chains, plot_post, plot_trace, plot_run, plot_final = derive_plot_flags(args)

    # Objects
    orbit_obj = Orbit()
    normal = "normal" in prior_type
    unit_cube = "unit-cube" in prior_type

    fit_obj = Fit(sampling_package=args.sampler, LOM_enabled=LOM_enabled, ndim=ndim, labels=labels, normal=normal, unit_cube=unit_cube)
    fit_obj.plot_chains = plot_chains
    vbm = VBMicrolensing(); vbm.a1 = 0.36

    if not os.path.exists(path + "posteriors/"):
        os.mkdir(path + "posteriors/")

    # Apply observatory group selection if requested
    if args.obs_group is not None:
        try:
            data_obj = Data()
            # Load config for the provided data path
            data_obj._load_config(path)
            # Ensure prm_file is set without interactive prompts
            prm_path = None
            if data_obj._config.get('prm_file') and os.path.exists(data_obj._config['prm_file']):
                prm_path = data_obj._config['prm_file']
            else:
                # Search for a .prm file in the data path
                candidates = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.prm')]
                if len(candidates) == 1:
                    prm_path = candidates[0]
                    data_obj._config['prm_file'] = prm_path
                    data_obj._save_config()
                elif len(candidates) == 0:
                    raise RuntimeError("--obs-group requires a parameter (.prm) file in the data directory, but none was found.")
                else:
                    raise RuntimeError(f"--obs-group requires a parameter (.prm) file, but multiple were found: {candidates}. Please set one in .gulls_config.json.")

            # Now set the observatory group
            data_obj.set_obs_group(args.obs_group)
            print(f"Using observatory group {args.obs_group}: {data_obj.obs_list}")
        except Exception as e:
            sys.exit(f"Failed to set observatory group {args.obs_group}: {e}")

    # Process events
    for i in range(total_events):
        fit_obj.current_event = None
        # Create Data object; preserve obs_list set above (if any)
        if 'data_obj' in locals() and isinstance(data_obj, Data) and getattr(data_obj, 'obs_list', None) is not None:
            # Reuse the configured Data object so obs_list and config persist
            pass
        else:
            data_obj = Data()

        if event_identifiers is not None:
            target_identifier = event_identifiers[i]
            try:
                event_name, truths_series, data = data_obj.load_event_by_identifier(path, target_identifier)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Skipping event {target_identifier}: {exc}")
                continue
        else:
            event_name, truths_series, data = data_obj.new_event(path, args.sort)
            if event_name is None:
                print(f"No more new events to process after {i} events. Exiting.")
                break
        truths = truths_series.to_dict()
        if 'params' in truths and isinstance(truths['params'], list):
            truths['params'] = np.array(truths['params'])

        print("\n\n\n\nevent_name =", event_name)
        print("---------------------------------------")
        print("truths =", truths)

        # Set true params for Fisher-informed priors
        fit_obj.true_params = truths["params"][:ndim]

        # Fisher setup for plotting and/or priors
        fit_obj.fisher_uncertainties_for_prior = None
        fit_obj.fisher_uncertainties_for_plotting = None
        fit_obj.fisher_covariance_for_plotting = None
        if args.use_fisher_prior and data_obj.model_parameter_uncertainties is not None:
            # Build label->sigma dict for robustness (handles extra flux params gracefully)
            base_model_labels = ["s","q","rho","u0","alpha","t0","tE","piEE","piEN"]
            sig = np.asarray(data_obj.model_parameter_uncertainties).reshape(-1)
            fisher_dict = {lbl: float(sig[i]) for i, lbl in enumerate(base_model_labels) if i < len(sig)}
            fit_obj.fisher_uncertainties_for_prior = fisher_dict
            fit_obj.fisher_uncertainties_for_plotting = fisher_dict
            fit_obj.fisher_covariance_for_plotting = data_obj.model_covariance
        elif data_obj.model_parameter_uncertainties is not None:
            fit_obj.fisher_covariance_for_plotting = data_obj.model_covariance
            base_model_labels = ["s","q","rho","u0","alpha","t0","tE","piEE","piEN"]
            sig = np.asarray(data_obj.model_parameter_uncertainties).reshape(-1)
            fisher_dict = {lbl: float(sig[i]) for i, lbl in enumerate(base_model_labels) if i < len(sig)}
            fit_obj.fisher_uncertainties_for_plotting = fisher_dict

        # Repackage data
        piE = np.array([truths["piEN"], truths["piEE"]])
        t0 = truths["params"][5]
        tE = truths["params"][6]
        tu_data, epochs, t_data, f_true, f_err_true, f_measured, f_err_measured = {}, {}, {}, {}, {}, {}, {}
        for obs in data.keys():
            tu_data[obs] = data[obs][3:5, :].T
            epochs[obs] = data[obs][0, :]
            f_measured[obs] = data[obs][1, :]  # measured_relative_flux
            f_err_measured[obs] = data[obs][2, :]  # measured_relative_flux_error
            f_true[obs] = data[obs][5, :]  # true_relative_flux
            f_err_true[obs] = data[obs][6, :]  # true_relative_flux_error
            t_data[obs] = data[obs][0, :]

        parallax_obj = Parallax(truths["ra_deg"], truths["dec_deg"], orbit_obj,
                                truths["tcroin"], tu_data, piE, epochs)
        parallax_obj.update_piE_NE(truths["piEN"], truths["piEE"])

        event_t0 = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, truths["t0lens1"], gamma=data_obj.gamma, LOM_enabled=LOM_enabled, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
        event_tc = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, truths["tcroin"], gamma=data_obj.gamma, LOM_enabled=LOM_enabled, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
        s, q, u0, alpha = truths["params"][0], truths["params"][1], truths["params"][3], truths["params"][4]
        tc_calc = event_tc.croin(t0, u0, s, q, alpha, tE)
        event_tref = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, tc_calc, gamma=data_obj.gamma, LOM_enabled=LOM_enabled, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)

        # Choose tref by chi^2
        chi2_ew_t0, _ = fit_obj.get_chi2(event_t0, truths["params"]) 
        chi2_ew_tc, _ = fit_obj.get_chi2(event_tc, truths["params"]) 
        chi2_ew_tref, _ = fit_obj.get_chi2(event_tref, truths["params"], measured_flux=False)
        
        # FAIL FAST: If magnification calculations failed with truth values, something is seriously wrong
        if chi2_ew_t0 is None or chi2_ew_tc is None or chi2_ew_tref is None:
            print("FATAL ERROR: Magnification calculations failed with truth values!")
            print("This indicates a serious problem:")
            print("  - VBMicrolensing/VBBinaryLensing libraries missing or broken")
            print("  - Truth parameter values are invalid/unphysical") 
            print("  - System configuration error")
            print("The code cannot proceed without working magnification calculations.")
            sys.exit(1)
        
        tmin = np.min([t0 - 2.0 * tE, tc_calc - 2.0 * tE]); tmax = np.max([t0 + 2.0 * tE, tc_calc + 2.0 * tE])
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))
        chi2_list = [np.sum(chi2_ew_t0[0][points]), np.sum(chi2_ew_tc[0][points]), np.sum(chi2_ew_tref[0][points])]
        tref_list = [truths["t0lens1"], truths["tcroin"], tc_calc]
        fit_tref = tref_list[int(np.argmin(chi2_list))]

        # ------------------------------------------------------------------
        # INITIAL FIGURES (restored from legacy script) BEFORE CROPPING
        # ------------------------------------------------------------------
        if plot_initial:
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
                base_colours = ["green", "red", "orange", "blue", "purple", "yellow"]
                default_labels = {0: "F146", 1: "F062", 2: "F087", 3: "F184", 4: "F213", 5: "F106"}
                ordered_obs = sorted(list(data.keys()))
                colour_map = {obs: base_colours[i % len(base_colours)] for i, obs in enumerate(ordered_obs)}
                label_map = {obs: default_labels.get(obs, f"Obs{obs}") for obs in ordered_obs}
                tt = np.linspace(tmin, tmax, 4000)

                for obs in ordered_obs:
                    A = event_t0.get_magnification(t_data[obs], obs)
                    
                    # Plot true flux (what you were using before) - solid color
                    fs_obs_true, fb_obs_true = fit_obj.get_fluxes(A, f_true[obs], f_err_true[obs] ** 2)
                    ax1.plot(t_data[obs], (f_true[obs] - fb_obs_true) / fs_obs_true, '.', 
                             color=colour_map[obs], label=f'{label_map[obs]} (true)', alpha=0.8, zorder=1)
                    residuals_true = f_true[obs] - (A * fs_obs_true + fb_obs_true)
                    ax2.plot(t_data[obs], residuals_true, '.', color=colour_map[obs], alpha=0.8, zorder=1)
                    
                    # Plot measured flux - same color but lower alpha
                    fs_obs_meas, fb_obs_meas = fit_obj.get_fluxes(A, f_measured[obs], f_err_measured[obs] ** 2)
                    ax1.plot(t_data[obs], (f_measured[obs] - fb_obs_meas) / fs_obs_meas, '.', 
                             color=colour_map[obs], label=f'{label_map[obs]} (measured)', alpha=0.4, zorder=0)
                    residuals_meas = f_measured[obs] - (A * fs_obs_meas + fb_obs_meas)
                    ax2.plot(t_data[obs], residuals_meas, '.', color=colour_map[obs], alpha=0.4, zorder=0)

                # Models at t0, tc, and calculated tref
                ax1.plot(tt, event_tc.get_magnification(tt, 0), '-', color='cyan', label=f"$t_c$={event_tc.t_ref:.1f}", lw=1, alpha=0.75)
                ax1.plot(tt, event_t0.get_magnification(tt, 0), '-', color='blue', label=f"$t_0$={event_t0.t_ref:.1f}", lw=1, alpha=0.75)
                ax1.plot(tt, event_tref.get_magnification(tt, 0), '-', color='purple', label=f"$t_c,calc$={event_tref.t_ref:.1f}", lw=1, alpha=0.75)
                ax1.axvline(x=fit_tref, color='orange', linestyle='-', alpha=0.25, zorder=0, linewidth=4)

                ax1.set_xlim(tmin, tmax)
                ax1.set_ylabel('Magnification')
                title_str = build_plot_titles(LOM_enabled)
                ax1.set_title(title_str % tuple(truths['params'][:ndim]))
                ax1.legend()
                ax2.set_ylabel('Residuals (flux - model)')
                ax2.set_xlabel('BJD')
                fig.tight_layout()
                plt.savefig(path + f"posteriors/{event_name}_truths_lightcurve.png", dpi=200)
                plt.close(fig)

                # Caustic plot (center on source at t_ref, show trajectory, fixed +/-3 bounds)
                fig = plt.figure()
                axc = plt.gca()

                # Ensure trajectory diagnostics exist for event_tref by evaluating magnification on data epochs
                try:
                    for obs in ordered_obs:
                        _ = event_tref.get_magnification(t_data[obs], obs)
                except Exception:
                    pass

                # Choose an observatory to anchor the center (first available)
                first_obs = ordered_obs[0]  # this is the Roman wide filter. All observations are from Roman
                t_arr = t_data[first_obs]
                # Find closest time index to t_ref
                idx_center = int(np.argmin(np.abs(t_arr - event_tref.t_ref)))
                
                # Source position at (approx) t_ref in COM, rotated frame
                if first_obs in event_tref.traj_parallax_dalpha_u1:
                    x0 = event_tref.traj_parallax_dalpha_u1[first_obs][idx_center]
                else:
                    x0 = 0.0
                    print("Warning: No trajectory data for dAlpha_u1; defaulting x0=0.0")
                if first_obs in event_tref.traj_parallax_dalpha_u2:
                    y0 = event_tref.traj_parallax_dalpha_u2[first_obs][idx_center]
                else:
                    y0 = 0.0
                    print("Warning: No trajectory data for dAlpha_u2; defaulting y0=0.0")

                # Plot trajectory for each observatory if available
                u1_arr = event_tref.traj_parallax_dalpha_u1[first_obs]
                u2_arr = event_tref.traj_parallax_dalpha_u2[first_obs]
                if u1_arr is not None and u2_arr is not None:
                    axc.plot(u1_arr, u2_arr, '-', alpha=0.4, lw=1.0)

                # Mark the t_ref point
                axc.plot(x0, y0, marker='*', color='k', ms=2, zorder=5)
                
                # Draw caustics using separation near t_ref
                s_use = float(truths['params'][0])
                q_use = float(truths['params'][1])
                caustics = vbm.Caustics(s_use, q_use)
                for closed in caustics:
                    axc.plot(closed[0], closed[1], '-', color='blue', ms=1, alpha=0.7)

                # Lens positions (COM frame at t_ref)
                axc.plot(event_tref.lens1_0[0], event_tref.lens1_0[1], 'o', ms=6, color='red')
                axc.plot(event_tref.lens2_0[0], event_tref.lens2_0[1], 'o', ms=6*q_use, color='red')

                # Center and bounds
                axc.set_aspect('equal', adjustable='box')
                axc.set_xlim(x0 - 5.0, x0 + 5.0)
                axc.set_ylim(y0 - 5.0, y0 + 5.0)
                axc.grid(True, alpha=0.3)
                plt.savefig(
                    path + f"posteriors/{event_name}_truths_caustic.png", 
                    dpi=200, 
                    bbox_inches='tight'
                )
                plt.close(fig)
                
            except Exception as e:
                print(
                    f"Error occurred while plotting initial plots for event {event_name}: \n"
                    f"{e}"
                )

        # Crop data around event
        t0_win, tE_win = truths["params"][5], truths["params"][6]
        tmin_fit = min(t0_win - 1.5 * tE_win, tc_calc - 1.5 * tE_win)
        tmax_fit = max(t0_win + 1.5 * tE_win, tc_calc + 1.5 * tE_win)
        data_cropped = {}
        for obs_key in data.keys():
            current_t = data[obs_key][0, :]
            pts = np.where((current_t > tmin_fit) & (current_t < tmax_fit))
            data_cropped[obs_key] = data[obs_key].T[pts].T

        event_fit = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=LOM_enabled, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)

        # Save sampling parameters to .prm file before starting
        # add normal and unit_cube to the fit object init

        save_run_parameters(args, event_name, path, truths, ndim, labels, 
                           prange_linear, prange_log, prior_type, start_time, fit_obj, data_obj.gamma)

        # Sampler setup
        print(f"\nSampling Posterior using {args.sampler}")
        normal = (prior_type in ["normal", "normal-unit-cube"])  # normal vs uniform priors in physical space
        nl, mi, stepi = 200, args.n_samples, args.n_step

        if args.sampler == "emcee":
            # Decide lnp and initial positions
            if fit_obj.unit_cube:
                lnp = fit_obj.lnprob_transform
                initial_pos = np.ones((nl, ndim)) * 0.5 + 1e-10 * np.random.rand(nl, ndim)
            else:
                lnp = fit_obj.lnprob
                # lnprob expects log-transformed parameters for s, q, rho, tE (and period if LOM)
                initial_pos = np.tile(truths["params"][:ndim], (nl, 1))
                log_indices = [0,1,2,6,11] if LOM_enabled else [0,1,2,6]
                
                # Transform log parameters to log space
                for j in log_indices:
                    if j < initial_pos.shape[1]:  # Safety check
                        initial_pos[:, j] = np.log10(initial_pos[:, j])
                
                # Add scatter in the appropriate space
                scatter = 1e-4
                for j in range(ndim):
                    if j in log_indices:
                        # Scatter in log space (additive)
                        initial_pos[:, j] += scatter * np.random.randn(nl)
                    else:
                        # Scatter in linear space (additive) 
                        initial_pos[:, j] += scatter * (p_unc[j] if j < len(p_unc) else 1.0) * np.random.randn(nl)

            if adaptive_burnin:
                state, p_unc_out, prange_linear, prange_log = fit_obj.run_burnin(
                    nl, ndim, args.burnin_stepi, lnp, initial_pos, event_fit, truths,
                    prange_linear, prange_log, p_unc, normal, max_steps=args.burnin_max_steps,
                    threads=args.threads, event_name=event_name, path=path, labels=labels,
                    min_steps=args.burnin_min_steps, fisher_uncertainties_for_plotting=fit_obj.fisher_uncertainties_for_plotting,
                    plot_chains=plot_chains,
                    show_progress=False
                )
            else:
                state = initial_pos

            sampler = fit_obj.run_emcee(
                nl, ndim, stepi, mi, lnp, state, event_fit, truths, prange_linear, prange_log, normal,
                threads=args.threads, event_name=event_name, path=path, labels=labels,
                fisher_uncertainties_for_plotting=fit_obj.fisher_uncertainties_for_plotting,
                fisher_uncertainties_for_prior=fit_obj.fisher_uncertainties_for_prior,
                plot_chains=plot_chains,
                show_progress=False
            )

            flat_chain = sampler.get_chain(flat=True)
        else:
            sampler = fit_obj.run_dynesty(
                event_fit, event_name, ndim, path, truths, prange_linear, prange_log, normal,
                fit_obj.fisher_uncertainties_for_prior
            )
            flat_chain = sampler.results.samples

        # Convert to physical space if needed
        if fit_obj.unit_cube:
            samples_phys = fit_obj.prior_transform(flat_chain, truths["params"][:ndim], prange_linear, prange_log,
                                                   normal=normal, fisher_uncertainties=fit_obj.fisher_uncertainties_for_prior)
        else:
            samples_phys = flat_chain

        np.save(path + f"posteriors/{event_name}_post_samples.npy", samples_phys)

        def _save_vbm_array(label, data_list):
            """Save VBM failure logs. Empty array means zero failures (good info to keep)."""
            outfile = path + f"posteriors/{event_name}_{label}.npy"
            arr = np.asarray(data_list, dtype=float) if data_list else np.array([], dtype=float)
            np.save(outfile, arr)

        if hasattr(event_fit, 'vbm_fault_params'):
            _save_vbm_array('vbm_faults', event_fit.vbm_fault_params)
        if hasattr(event_fit, 'vbm_timeout_params'):
            _save_vbm_array('vbm_timeouts', event_fit.vbm_timeout_params)

        with open(path + f"posteriors/{event_name}end_truths.pkl", "wb") as f:
            pickle.dump(truths, f)

        # Final plots (optional)
        if plot_post:
            # Prepare samples for corner plot regardless of plot_chains setting
            flat_chain_post = sampler.get_chain(flat=True) if args.sampler == "emcee" else sampler.results.samples

            if fit_obj.unit_cube:
                # Unit-cube case: transform from unit-cube to physical space
                samples_for_corner = fit_obj.prior_transform(flat_chain_post, truths["params"][:ndim], prange_linear, prange_log,
                                        normal=normal, fisher_uncertainties=fit_obj.fisher_uncertainties_for_prior)
            else:
                # Regular case: chain is in log space for some parameters, need to transform to physical space
                samples_for_corner = flat_chain_post.copy()
                log_indices = [0,1,2,6,11] if LOM_enabled else [0,1,2,6]
                
                # Transform log parameters back to linear space
                for j in log_indices:
                    if j < samples_for_corner.shape[1]:  # Safety check
                        samples_for_corner[:, j] = 10**(samples_for_corner[:, j])
            
            log_param_names = ["s","q","rho","tE","period"] if LOM_enabled else ["s","q","rho","tE"]   
            fit_obj.corner_post(samples_for_corner, event_name, path, truths,
                                fisher_covariance=fit_obj.fisher_covariance_for_plotting,
                                fisher_uncertainties=fit_obj.fisher_uncertainties_for_plotting,
                                log_param_names=log_param_names)
        if plot_trace and hasattr(fit_obj, 'traceplot') and args.sampler == "dynesty":
            fit_obj.traceplot(sampler, event_name, path, truths)
        if plot_run and hasattr(fit_obj, 'runplot') and args.sampler == "dynesty":
            fit_obj.runplot(sampler, event_name, path)

        # Final lightcurve plots with posterior samples
        if plot_final:
            try:
                print("Generating final lightcurve plots with posterior samples...")
                
                # Get samples in physical space (already computed above)
                if args.sampler == "emcee":
                    # Remove burn-in from chain
                    burnin_remove = max(100, args.burnin_min_steps // 2)  # Remove at least some burn-in
                    chain_no_burnin = sampler.get_chain(discard=burnin_remove, flat=True)
                    
                    print(f"Debug: chain_no_burnin shape: {chain_no_burnin.shape}")
                    
                    if chain_no_burnin.size == 0:
                        print("Warning: No samples after burn-in removal, skipping final plots")
                        continue
                    
                    if fit_obj.unit_cube:
                        samples_final = fit_obj.prior_transform(chain_no_burnin, truths["params"][:ndim], 
                                                               prange_linear, prange_log, normal=normal, 
                                                               fisher_uncertainties=fit_obj.fisher_uncertainties_for_prior)
                    else:
                        samples_final = chain_no_burnin.copy()
                        log_indices = [0,1,2,6,11] if LOM_enabled else [0,1,2,6]
                        
                        # Transform log parameters back to linear space
                        for j in log_indices:
                            if j < samples_final.shape[1]:
                                samples_final[:, j] = 10**(samples_final[:, j])
                else:
                    # Dynesty samples are already in physical space
                    samples_final = samples_phys
                
                print(f"Debug: samples_final shape: {samples_final.shape}")
                
                if samples_final.size == 0:
                    print("Warning: No final samples available, skipping final plots")
                    raise ValueError("No samples for plotting")
                
                # Get 50th percentile sample
                median_params = np.percentile(samples_final, 50, axis=0)
                print(f"Debug: median_params shape: {median_params.shape}")
                
                # Select some random samples for transparent overlay (5-10 samples)
                n_samples_plot = min(10, len(samples_final))
                if n_samples_plot > 0:
                    random_indices = np.random.choice(len(samples_final), n_samples_plot, replace=False)
                    random_samples = samples_final[random_indices]
                    print(f"Debug: selected {len(random_samples)} random samples")
                else:
                    random_samples = []
                    print("Debug: no random samples selected")
                    raise ValueError("No samples available for plotting")
                
                # Create the final lightcurve plot
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
                base_colours = ["orange", "red", "green", "purple", "cyan", "magenta", "brown", "olive"]
                default_labels = {0: "W146", 1: "Z087", 2: "K213"}
                ordered_obs = sorted(list(data_cropped.keys()))
                colour_map = {obs: base_colours[i % len(base_colours)] for i, obs in enumerate(ordered_obs)}
                label_map = {obs: default_labels.get(obs, f"Obs{obs}") for obs in ordered_obs}
                
                # Plot data
                for obs in ordered_obs:
                    t_obs = data_cropped[obs][0, :]
                    f_obs = data_cropped[obs][1, :]  # observed relative flux: F = fs*A + (1-fs)
                    ferr_obs = data_cropped[obs][2, :]  # observed relative flux error
                    
                    # Convert relative flux to magnification using median model
                    # F = fs*A + (1-fs), so A = (F - (1-fs)) / fs = (F - fb) / fs
                    temp_params = median_params.copy()
                    if LOM_enabled and len(temp_params) >= 12:
                        temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=True, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                    else:
                        temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=False, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                    
                    temp_event.set_params(temp_params)
                    A_model = temp_event.get_magnification(t_obs, obs)
                    
                    if A_model is not None:
                        # Get fitted flux parameters: F = fs*A + fb
                        fs, fb = fit_obj.get_fluxes(A_model, f_obs, ferr_obs**2)
                        
                        # Convert observed relative flux to magnification
                        if fs > 1e-6:  # Avoid division by very small numbers
                            A_obs = (f_obs - fb) / fs
                            Aerr_obs = ferr_obs / fs  # Error propagation
                        else:
                            # Fallback if fs is too small (pure blend case)
                            A_obs = f_obs
                            Aerr_obs = ferr_obs
                        
                        # Plot converted magnification
                        ax1.errorbar(t_obs, A_obs, yerr=Aerr_obs, 
                                   fmt='.', color=colour_map[obs], label=label_map[obs], alpha=0.7, zorder=2)
                        
                        # Plot residuals (observed magnification - model magnification)
                        residuals = A_obs - A_model
                        ax2.errorbar(t_obs, residuals, yerr=Aerr_obs, fmt='.', color=colour_map[obs], alpha=0.7, zorder=2)
                    else:
                        # Fallback if magnification calculation fails
                        ax1.errorbar(t_obs, f_obs, yerr=ferr_obs, 
                                   fmt='.', color=colour_map[obs], label=label_map[obs], alpha=0.7, zorder=2)
                
                # Plot models - create fine time grid
                t_fine = np.linspace(tmin_fit, tmax_fit, 2000)
                
                # Median model
                if LOM_enabled and len(median_params) >= 12:
                    temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=True, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                else:
                    temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=False, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                temp_event.set_params(median_params)
                A_fine_median = temp_event.get_magnification(t_fine, 0)
                ax1.plot(t_fine, A_fine_median, '-', color='black', linewidth=2, label='Median posterior', zorder=3)
                
                # Random posterior samples (transparent)
                for j, sample_params in enumerate(random_samples):
                    if LOM_enabled and len(sample_params) >= 12:
                        temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=True, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                    else:
                        temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=False, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                    temp_event.set_params(sample_params)
                    A_fine_sample = temp_event.get_magnification(t_fine, 0)
                    label_str = 'Posterior samples' if j == 0 else None
                    ax1.plot(t_fine, A_fine_sample, '-', color='gray', alpha=0.2, linewidth=1, 
                            label=label_str, zorder=1)
                
                # Truth model for comparison
                if LOM_enabled and len(truths["params"]) >= 12:
                    temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=True, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                else:
                    temp_event = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, gamma=data_obj.gamma, LOM_enabled=False, eps=data_obj.vbm_rel_tol, vbm_timeout=data_obj.vbm_timeout)
                temp_event.set_params(truths["params"][:ndim])
                A_fine_truth = temp_event.get_magnification(t_fine, 0)
                ax1.plot(t_fine, A_fine_truth, '--', color='green', linewidth=2, label='Truth', zorder=3)
                
                ax1.set_ylabel('Magnification')
                ax1.set_title(f'Final Lightcurve Fit - {event_name}')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                
                ax2.set_ylabel('Residuals')
                ax2.set_xlabel('BJD')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(path + f"posteriors/{event_name}_final_lightcurve.png", dpi=200, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved final lightcurve plot: {path}posteriors/{event_name}_final_lightcurve.png")
                
            except Exception as e:
                import traceback
                print(f"Warning: final lightcurve plotting failed for {event_name}: {e}")
                print("Traceback:")
                traceback.print_exc()

        print(f"Event {i} ({event_name}) is done")
        if not os.path.exists(path + "emcee_complete.txt"):
            np.savetxt(path + "emcee_complete.txt", np.array([]), fmt="%s")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            complete_list = np.atleast_1d(np.loadtxt(path + "emcee_complete.txt", dtype=str))
            complete_list = np.hstack([complete_list, event_name])
            np.savetxt(path + "emcee_complete.txt", complete_list, fmt="%s")

    end_time = time.time()
    print("\n\n--- Timing Summary ---")
    print("Total time =", end_time - start_time)
    print("--------------------------------\n\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)
