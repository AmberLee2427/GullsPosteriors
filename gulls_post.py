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
import matplotlib.pyplot as plt

from Data import Data
from Parallax import Parallax
from Event import Event
from Fit import Fit
from Orbit import Orbit
from VBMicrolensing import VBMicrolensing


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run posterior sampling on gull events")
    p.add_argument("nevents", type=int, help="Number of events to process")
    p.add_argument("path", help="Directory containing data challenge files")
    p.add_argument("-s", dest="sampler", choices=["emcee", "dynesty"], default="emcee")
    p.add_argument("-t", dest="threads", type=int, default=1, help="Number of threads (emcee)")
    p.add_argument("-sort", dest="sort", default="alphanumeric")
    p.add_argument("-noLOM", dest="no_lom", action="store_true", help="Disable lens orbit motion")
    p.add_argument("-fp", dest="use_fisher_prior", action="store_true", help="Use Fisher uncertainties for priors")
    p.add_argument("-adapt", dest="adaptive_burnin", action="store_true")
    p.add_argument("-prior", dest="prior", choices=["normal", "uniform", "uniform-unit-cube", "normal-unit-cube"],
                   help="Prior type")
    p.add_argument("-n", dest="n_samples", type=int, default=1000)
    p.add_argument("-nbimin", dest="burnin_min_steps", type=int, default=500)
    p.add_argument("-nbimax", dest="burnin_max_steps", type=int, default=1000)
    p.add_argument("-nbistep", dest="burnin_stepi", type=int, default=200)
    p.add_argument("-nstep", dest="n_step", type=int, default=100)
    p.add_argument("-f", dest="plots", default="ictpf", help="Plot flags: i(ni), c, t, p, f; use n to disable all")
    return p.parse_args(argv)


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

    LOM_enabled = not args.no_lom
    print("Lens Orbit Motion (LOM) is {}.".format("ENABLED" if LOM_enabled else "DISABLED"))
    print("{} Fisher uncertainties to inform prior ranges.".format(
        "Using" if args.use_fisher_prior else "NOT using"))

    ndim, labels, p_unc, prange_log, prange_linear = compute_prior_setup(LOM_enabled)

    adaptive_burnin = args.adaptive_burnin
    if adaptive_burnin and args.use_fisher_prior:
        sys.exit("Fisher informed prior ranges not supported with adaptive burn-in.")

    prior_type = choose_prior_type(args)

    plot_initial, plot_chain, plot_post, plot_trace, plot_run, plot_final = derive_plot_flags(args)

    # Objects
    orbit_obj = Orbit()
    fit_obj = Fit(sampling_package=args.sampler, LOM_enabled=LOM_enabled, ndim=ndim, labels=labels)
    vbm = VBMicrolensing(); vbm.a1 = 0.36

    if not os.path.exists(path + "posteriors/"):
        os.mkdir(path + "posteriors/")

    # Process events
    for i in range(args.nevents):
        fit_obj.current_event = None
        data_obj = Data()
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

        # Fisher setup for plotting and/or priors
        fit_obj.fisher_uncertainties_for_prior = None
        fit_obj.fisher_uncertainties_for_plotting = None
        fit_obj.fisher_covariance_for_plotting = None
        if args.use_fisher_prior and data_obj.model_parameter_uncertainties is not None:
            fit_obj.fisher_uncertainties_for_prior = data_obj.model_parameter_uncertainties
            fit_obj.fisher_uncertainties_for_plotting = data_obj.model_parameter_uncertainties
            fit_obj.fisher_covariance_for_plotting = data_obj.model_covariance
        elif data_obj.model_parameter_uncertainties is not None:
            fit_obj.fisher_covariance_for_plotting = data_obj.model_covariance
            fit_obj.fisher_uncertainties_for_plotting = data_obj.model_parameter_uncertainties

        # Repackage data
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

        parallax_obj = Parallax(truths["ra_deg"], truths["dec_deg"], orbit_obj,
                                truths["tcroin"], tu_data, piE, epochs)
        parallax_obj.update_piE_NE(truths["piEN"], truths["piEE"])

        event_t0 = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, truths["t0lens1"], LOM_enabled=LOM_enabled)
        event_tc = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, truths["tcroin"], LOM_enabled=LOM_enabled)
        s, q, u0, alpha = truths["params"][0], truths["params"][1], truths["params"][3], truths["params"][4]
        tc_calc = event_tc.croin(t0, u0, s, q, alpha, tE)
        event_tref = Event(parallax_obj, orbit_obj, data, truths, data_obj.sim_time0, tc_calc, LOM_enabled=LOM_enabled)

        # Choose tref by chi^2
        chi2_ew_t0, _ = fit_obj.get_chi2(event_t0, truths["params"]) 
        chi2_ew_tc, _ = fit_obj.get_chi2(event_tc, truths["params"]) 
        chi2_ew_tref, _ = fit_obj.get_chi2(event_tref, truths["params"], measured_flux=False)
        tmin = np.min([t0 - 2.0 * tE, tc_calc - 2.0 * tE]); tmax = np.max([t0 + 2.0 * tE, tc_calc + 2.0 * tE])
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))
        chi2_list = [np.sum(chi2_ew_t0[0][points]), np.sum(chi2_ew_tc[0][points]), np.sum(chi2_ew_tref[0][points])]
        tref_list = [truths["t0lens1"], truths["tcroin"], tc_calc]
        fit_tref = tref_list[int(np.argmin(chi2_list))]

        # Crop data around event
        t0_win, tE_win = truths["params"][5], truths["params"][6]
        tmin_fit = min(t0_win - 1.5 * tE_win, tc_calc - 1.5 * tE_win)
        tmax_fit = max(t0_win + 1.5 * tE_win, tc_calc + 1.5 * tE_win)
        data_cropped = {}
        for obs_key in data.keys():
            current_t = data[obs_key][0, :]
            pts = np.where((current_t > tmin_fit) & (current_t < tmax_fit))
            data_cropped[obs_key] = data[obs_key].T[pts].T

        event_fit = Event(parallax_obj, orbit_obj, data_cropped, truths, data_obj.sim_time0, fit_tref, LOM_enabled=LOM_enabled)

        # Sampler setup
        print(f"\nSampling Posterior using {args.sampler}")
        normal = (prior_type in ["normal", "normal-unit-cube"])  # normal vs uniform priors in physical space
        nl, mi, stepi = 200, args.n_samples, args.n_step

        if args.sampler == "emcee":
            # Decide lnp and initial positions
            if "unit-cube" in prior_type:
                lnp = fit_obj.lnprob_transform
                initial_pos = np.ones((nl, ndim)) * 0.5 + 1e-10 * np.random.rand(nl, ndim)
            else:
                lnp = fit_obj.lnprob
                initial_pos = np.tile(truths["params"][:ndim], (nl, 1))
                log_indices = [0,1,2,6,11] if LOM_enabled else [0,1,2,6]
                scatter = 1e-4
                for j in range(ndim):
                    if j in log_indices:
                        initial_pos[:, j] *= 10 ** (scatter * np.random.randn(nl))
                    else:
                        initial_pos[:, j] += scatter * (p_unc[j] if j < len(p_unc) else 1.0) * np.random.randn(nl)

            if adaptive_burnin:
                state, p_unc_out, prange_linear, prange_log = fit_obj.run_burnin(
                    nl, ndim, args.burnin_stepi, lnp, initial_pos, event_fit, truths,
                    prange_linear, prange_log, p_unc, normal, max_steps=args.burnin_max_steps,
                    threads=args.threads, event_name=event_name, path=path, labels=labels,
                    min_steps=args.burnin_min_steps, fisher_uncertainties_for_plotting=fit_obj.fisher_uncertainties_for_plotting,
                )
            else:
                state = initial_pos

            sampler = fit_obj.run_emcee(
                nl, ndim, stepi, mi, lnp, state, event_fit, truths, prange_linear, prange_log, normal,
                threads=args.threads, event_name=event_name, path=path, labels=labels,
                fisher_uncertainties_for_plotting=fit_obj.fisher_uncertainties_for_plotting,
                fisher_uncertainties_for_prior=fit_obj.fisher_uncertainties_for_prior,
            )

            flat_chain = sampler.get_chain(flat=True)
        else:
            sampler = fit_obj.run_dynesty(
                event_fit, event_name, ndim, path, truths, prange_linear, prange_log, normal,
                fit_obj.fisher_uncertainties_for_prior
            )
            flat_chain = sampler.results.samples

        # Convert to physical space if needed
        if "unit-cube" in prior_type:
            samples_phys = fit_obj.prior_transform(flat_chain, truths["params"][:ndim], prange_linear, prange_log,
                                                   normal=normal, fisher_uncertainties=fit_obj.fisher_uncertainties_for_prior)
        else:
            samples_phys = flat_chain

        np.save(path + f"posteriors/{event_name}_post_samples.npy", samples_phys)
        with open(path + f"posteriors/{event_name}end_truths.pkl", "wb") as f:
            pickle.dump(truths, f)

        # Final plots (optional)
        if plot_post:
            flat_chain_post = sampler.get_chain(flat=True) if args.sampler == "emcee" else sampler.results.samples
            samples_for_corner = (fit_obj.prior_transform(flat_chain_post, truths["params"][:ndim], prange_linear, prange_log,
                                    normal=normal, fisher_uncertainties=fit_obj.fisher_uncertainties_for_prior)
                                  if "unit-cube" in prior_type else flat_chain_post)
            log_param_names = ["s","q","rho","tE","period"] if LOM_enabled else ["s","q","rho","tE"]
            fit_obj.corner_post(samples_for_corner, event_name, path, truths,
                                fisher_covariance=fit_obj.fisher_covariance_for_plotting,
                                fisher_uncertainties=fit_obj.fisher_uncertainties_for_plotting,
                                log_param_names=log_param_names)
        if plot_trace and hasattr(fit_obj, 'traceplot') and args.sampler == "dynesty":
            fit_obj.traceplot(sampler, event_name, path, truths)
        if plot_run and hasattr(fit_obj, 'runplot') and args.sampler == "dynesty":
            fit_obj.runplot(sampler, event_name, path)

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

