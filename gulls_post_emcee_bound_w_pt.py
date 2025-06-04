"""Command line interface to run posterior sampling on gull events."""

# import multiprocessing
# multiprocessing.set_start_method('fork', force=True)
import warnings

# warnings.filterwarnings(
#    "ignore",
#    message=(
#        "resource_tracker: There appear to be .* leaked semaphore objects"
#    ),
# )

import os
import sys
import numpy as np
import pickle
import corner
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

if __name__ == "__main__":

    start_time = time.time()
    print("Start time = ", start_time)

    # --- Script Options ---
    nevents = int(sys.argv[1])

    if "-s" in sys.argv:
        sampler_index = sys.argv.index("-s") + 1
        sampling_package = sys.argv[sampler_index]
    else:
        sampling_package = "emcee"

    if "-t" in sys.argv:
        threads_index = sys.argv.index("-t") + 1
        threads = int(sys.argv[threads_index])
        if threads == 0:
            threads = mp.cpu_count()
    else:
        threads = 1
        pooling = False

    # --- Plotting Defaults ---
    plot_chain = True if sampling_package == "emcee" else False
    plot_trace = True if sampling_package == "dynesty" else False
    plot_post = True if sampling_package == "emcee" else False
    plot_run = True if sampling_package == "dynesty" else False
    plot_initial_figures = True
    plot_final_figures = True

    if "-f" in sys.argv:
        plot_index = sys.argv.index("-f") + 1
        plot_options = sys.argv[plot_index]
        if "n" in plot_options:
            plot_initial_figures = False
            plot_final_figures = False
            plot_chain = False
            plot_trace = False
            plot_post = False
            plot_run = False
        if "i" in plot_options:
            plot_initial_figures = True
        else:
            plot_initial_figures = False
        if "c" in plot_options:
            plot_chain = True
        else:
            plot_chain = False
        if "t" in plot_options:
            plot_trace = True
        else:
            plot_trace = False
        if "p" in plot_options:
            plot_post = True
        else:
            plot_post = False
        if "f" in plot_options:
            plot_final_figures = True
        if plot_final_figures:
            if sampling_package == "dynesty":
                plot_trace = True
                plot_run = True
            if sampling_package == "emcee":
                plot_post = True

    path = sys.argv[2]
    if path[-1] != "/":
        path = path + "/"

    if "-sort" in sys.argv:
        sort_index = sys.argv.index("-sort") + 1
        sort = sys.argv[sort_index]
    else:
        sort = "alphanumeric"

    # --- LOM Flag ---
    if "-noLOM" in sys.argv:
        LOM_enabled = False
        print("Lens Orbit Motion (LOM) is DISABLED.")
    else:
        LOM_enabled = True
        print("Lens Orbit Motion (LOM) is ENABLED.")

    # ==================================================================
    # CONDITIONAL PARAMETER AND PRIOR SETUP MOVED HERE (OUTSIDE THE LOOP)
    # ==================================================================
    if LOM_enabled:
        ndim = 12
        labels = [
            "s",
            "q",
            "rho",
            "u0",
            "alpha",
            "t0",
            "tE",
            "piEE",
            "piEN",
            "i",
            "phase",
            "period",
        ]
        p_unc = np.array(
            [
                0.05,
                0.05,
                0.1,
                0.1,
                0.05,
                0.5,
                2.5,
                2.5,
                5.0,
                np.pi / 2.0,
                np.pi / 2.0,
                0.2,
            ]
        )

        # Define ranges for 12 params
        # Log-space parameters
        p_unc_log_space = np.array([0.1, 0.1, 0.1, 0.05])  # s, q, rho, period
        prange_log = p_unc_log_space * 2.0

        # Linear-space parameters
        linear_indices = [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ]  # u0, alpha, t0, tE, piEE, piEN, i, phase
        p_unc_linear_space = p_unc[linear_indices]
        prange_linear = p_unc_linear_space * 2.0

    else:  # LOM is DISABLED
        ndim = 9
        labels = [
            "s",
            "q",
            "rho",
            "u0",
            "alpha",
            "t0",
            "tE",
            "piEE",
            "piEN",
        ]
        p_unc = np.array([0.05, 0.05, 0.1, 0.1, 0.05, 0.5, 2.5, 2.5, 5.0])

        # Define ranges for 9 params
        # Log-space parameters (no period)
        p_unc_log_space = np.array([0.1, 0.1, 0.1])  # s, q, rho
        prange_log = p_unc_log_space * 2.0

        # Linear-space parameters (no i, phase)
        linear_indices = [
            3,
            4,
            5,
            6,
            7,
            8,
        ]  # u0, alpha, t0, tE, piEE, piEN
        p_unc_linear_space = p_unc[linear_indices]
        prange_linear = p_unc_linear_space * 2.0
    # ==================================================================
    # END OF MOVED PARAMETER SETUP BLOCK
    # ==================================================================

    # --- Imports ---
    from Data import Data
    from Parallax import Parallax
    from Event import Event
    from Fit import Fit
    from Orbit import Orbit
    from Fit._dynesty import prior_transform
    from VBMicrolensing import VBMicrolensing

    # --- Initializations ---
    orbit_obj = Orbit()
    fit_obj = Fit(
        sampling_package=sampling_package,
        LOM_enabled=LOM_enabled,
        ndim=ndim,
        labels=labels,
    )
    vbm = VBMicrolensing()
    vbm.a1 = 0.36

    if not os.path.exists(path + "posteriors/"):
        os.mkdir(path + "posteriors/")

    # --- Main Event Loop ---
    for i in range(nevents):
        data_obj = Data()
        event_name, truths, data = data_obj.new_event(path, sort)

        print(f"\n\n\n\n\n\nevent_name = {event_name}")
        print("---------------------------------------")
        print("truths = ", truths)

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

        parallax_obj = Parallax(
            truths["ra_deg"],
            truths["dec_deg"],
            orbit_obj,
            truths["tcroin"],
            tu_data,
            piE,
            epochs,
        )
        parallax_obj.update_piE_NE(truths["piEN"], truths["piEE"])

        event_t0 = Event(
            parallax_obj,
            orbit_obj,
            data,
            truths,
            data_obj.sim_time0,
            truths["t0lens1"],
            LOM_enabled=LOM_enabled,
        )
        event_tc = Event(
            parallax_obj,
            orbit_obj,
            data,
            truths,
            data_obj.sim_time0,
            truths["tcroin"],
            LOM_enabled=LOM_enabled,
        )
        s, q, u0, alpha = (
            truths["params"][0],
            truths["params"][1],
            truths["params"][3],
            truths["params"][4],
        )
        tc_calc = event_tc.croin(t0, u0, s, q, alpha, tE)
        event_tref = Event(
            parallax_obj,
            orbit_obj,
            data,
            truths,
            data_obj.sim_time0,
            tc_calc,
            LOM_enabled=LOM_enabled,
        )

        # --- Chi2 comparison to find best t_ref ---
        chi2_ew_t0, chi2_t0 = fit_obj.get_chi2(event_t0, truths["params"])
        chi2_ew_tc, chi2_tc = fit_obj.get_chi2(event_tc, truths["params"])
        chi2_ew_tref, chi2_tref = fit_obj.get_chi2(
            event_tref,
            truths["params"],
        )

        tmin = np.min([t0 - 2.0 * tE, tc_calc - 2.0 * tE])
        tmax = np.max([t0 + 2.0 * tE, tc_calc + 2.0 * tE])
        points = np.where(np.logical_and(t_data[0] > tmin, t_data[0] < tmax))

        chi2_t0_0 = [np.sum(chi2_ew_t0[0]), np.sum(chi2_ew_t0[0][points])]
        chi2_tc_0 = [np.sum(chi2_ew_tc[0]), np.sum(chi2_ew_tc[0][points])]
        chi2_tref_0 = [
            np.sum(chi2_ew_tref[0]),
            np.sum(chi2_ew_tref[0][points]),
        ]

        # determining the tref used for LOM: fit_tref
        A_t0 = event_t0.get_magnification(t_data[0], 0)
        fs_t0, fb_t0 = fit_obj.get_fluxes(A_t0, f_true[0], f_err_true[0] ** 2)
        chi2_ew_t0_true = (
            (f_true[0] - (A_t0 * fs_t0 + fb_t0)) ** 2 / f_err_true[0] ** 2
        )[points]

        A_tc = event_tc.get_magnification(t_data[0], 0)
        fs_tc, fb_tc = fit_obj.get_fluxes(A_tc, f_true[0], f_err_true[0] ** 2)
        chi2_ew_tc_true = (
            (f_true[0] - (A_tc * fs_tc + fb_tc)) ** 2 / f_err_true[0] ** 2
        )[points]

        A_tref = event_tref.get_magnification(t_data[0], 0)
        fs_tref, fb_tref = fit_obj.get_fluxes(
            A_tref,
            f_true[0],
            f_err_true[0] ** 2,
        )
        chi2_ew_tref_true = (
            (f_true[0] - (A_tref * fs_tref + fb_tref)) ** 2
            / f_err_true[0] ** 2
        )[points]

        chi2_list = [
            np.sum(chi2_ew_t0_true),
            np.sum(chi2_ew_tc_true),
            np.sum(chi2_ew_tref_true),
        ]
        tref_list = [truths["t0lens1"], truths["tcroin"], tc_calc]
        fit_tref = tref_list[np.argmin(chi2_list)]

        end_preamble = time.time()
        print("Time to get data = ", end_preamble - start_time)

        # ==================================================================
        # INITIAL PLOTTING
        # ==================================================================
        if plot_initial_figures:
            # --- Initial Lightcurve Plot ---
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 2]},
            )
            colours = ["orange", "red", "green"]
            data_labels = ["W146", "Z087", "K213"]
            tt = np.linspace(tmin, tmax, 10000)

            for obs in event_tc.data.keys():
                A = event_tc.get_magnification(t_data[obs], obs)
                fs, fb = fit_obj.get_fluxes(A, data[obs][1], data[obs][2] ** 2)
                ax1.plot(
                    t_data[obs],
                    (data[obs][1] - fb) / fs,
                    ".",
                    color=colours[obs],
                    label=data_labels[obs],
                    alpha=0.5,
                    zorder=0,
                )

            ax1.plot(
                tt,
                event_tc.get_magnification(tt, 0),
                "-",
                color="cyan",
                label=f"$t_c$={event_tc.t_ref:.1f}",
                zorder=6,
                alpha=0.75,
                lw=1,
            )
            ax1.plot(
                tt,
                event_t0.get_magnification(tt, 0),
                "-",
                color="blue",
                label=f"$t_0$={event_t0.t_ref:.1f}",
                zorder=4,
                alpha=0.75,
                lw=1,
            )
            ax1.plot(
                tt,
                event_tref.get_magnification(tt, 0),
                "-",
                color="purple",
                label=f"$t_c,calc$={event_tref.t_ref:.1f}",
                zorder=2,
                alpha=0.75,
                lw=1,
            )
            ax1.axvline(
                x=fit_tref,
                color="orange",
                linestyle="-",
                alpha=0.25,
                zorder=0,
                linewidth=4,
            )

            ax1.set_xlim(tmin, tmax)
            ax1.set_ylabel("Magnification")
            if LOM_enabled:
                title_str = (
                    "s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, "(
                        "\ntE=%.2f, piEE=%.2f, piEN=%.2f, "
                        "i=%.2f, phase=%.2f, period=%.2f"
                    )
                )
            else:
                title_str = (
                    "s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, "
                    "\ntE=%.2f, piEE=%.2f, piEN=%.2f"
                )
            ax1.set_title(title_str % tuple(truths["params"][:ndim]))
            ax1.legend()
            plt.savefig(
                path + "posteriors/" + event_name + "_truths_lightcurve.png",
                dpi=300,
            )
            plt.close(fig)

            # --- LOM-Specific Plots ---
            if LOM_enabled:
                # Caustic Plot
                plt.figure()
                s_tc, _, _ = event_tc.projected_separation(
                    truths["params"][9],
                    truths["params"][11],
                    truths["tcroin"],
                    phase_offset=truths["params"][10],
                    t_start=truths["tcroin"],
                    a=truths["Planet_semimajoraxis"] / truths["rE"],
                )
                caustics_tc = vbm.Caustics(s_tc, truths["params"][1])
                for closed in caustics_tc:
                    plt.plot(
                        closed[0],
                        closed[1],
                        "-",
                        color="cyan",
                        ms=0.2,
                        zorder=0,
                        alpha=0.5,
                    )
                plt.plot(
                    event_tc.lens1_0[0],
                    event_tc.lens1_0[1],
                    "o",
                    ms=10,
                    color="red",
                    zorder=0,
                )
                plt.plot(
                    event_tc.lens2_0[0],
                    event_tc.lens2_0[1],
                    "o",
                    ms=10**q,
                    color="red",
                    zorder=0,
                )
                plt.plot(
                    event_tc.traj_parallax_dalpha_u1[0][points],
                    event_tc.traj_parallax_dalpha_u2[0][points],
                    "-",
                    color="cyan",
                    alpha=0.5,
                )
                plt.grid()
                plt.axis("equal")
                plt.legend()
                plt.savefig(
                    path + "posteriors/" + event_name + "_truths_caustic.png"
                )
                plt.close()

                # Delta s Plot
                plt.figure()
                plt.plot(
                    event_tc.tau[0],
                    event_tc.ss[0],
                    ".",
                    label="ss",
                    alpha=0.1,
                )
                plt.xlabel(r"$\tau$")
                plt.ylabel("s")
                plt.legend()
                plt.savefig(path + "posteriors/" + event_name + "_dsdtau.png")
                plt.close()

                # Delta alpha Plot
                plt.figure()
                plt.plot(
                    event_tc.tau[0],
                    event_tc.dalpha[0],
                    ".",
                    alpha=0.1,
                )
                plt.xlabel(r"$\tau$")
                plt.ylabel(r"d$\alpha$")
                plt.legend()
                plt.savefig(
                    path + "posteriors/" + event_name + "_dalphadtau.png"
                )
                plt.close()

        end_initial_figures = time.time()
        print(
            "Time to make initial figures = ",
            end_initial_figures - end_preamble,
        )

        # ==================================================================
        # SAMPLER SETUP AND RUN
        # ==================================================================
        print("\nSampling Posterior using emcee")
        print("--------------------------------")
        normal = True
        nl, mi, stepi = 200, 2000, 100
        initial_pos = np.ones((nl, ndim)) * 0.5 + 1e-10 * np.random.rand(
            nl, ndim
        )

        # Cropping data to near-event for fitting
        tmin_fit = fit_tref - 1.5 * tE
        tmax_fit = fit_tref + 1.5 * tE

        data_cropped = {}
        for obs_key in data.keys():
            # Get time data for the current observatory
            current_obs_t_data = data[obs_key][0, :]

            # Calculate cropping indices specifically for this
            # observatory's time data
            points_for_this_obs = np.where(
                np.logical_and(
                    current_obs_t_data > tmin_fit,
                    current_obs_t_data < tmax_fit,
                )
            )

            # Crop this observatory's data using its specific indices
            data_cropped[obs_key] = data[obs_key].T[points_for_this_obs].T

        event_fit = Event(
            parallax_obj,
            orbit_obj,
            data_cropped,
            truths,
            data_obj.sim_time0,
            fit_tref,
            LOM_enabled=LOM_enabled,
        )

        sampler = fit_obj.run_emcee(
            nl,
            ndim,
            stepi,
            mi,
            fit_obj.lnprob_transform,
            initial_pos,
            event_fit,
            truths["params"],
            prange_linear,
            prange_log,
            normal,
            threads=threads,
            event_name=event_name,
            path=path,
            labels=labels,
        )

        end_sampler = time.time()
        print(f"Time to run sampler= {end_sampler - end_initial_figures}")

        # ==================================================================
        # SAVE RESULTS AND MAKE FINAL PLOTS
        # ==================================================================
        flat_chain = sampler.get_chain(flat=True)
        samples_phys = fit_obj.prior_transform(
            flat_chain,
            truths["params"][:ndim],
            prange_linear,
            prange_log,
            normal=True,
        )
        np.save(
            path + "posteriors/" + event_name + "_post_samples.npy",
            samples_phys,
        )
        with open(
            path + "posteriors/" + event_name + "end_truths.pkl", "wb"
        ) as f:
            pickle.dump(truths, f)

        if plot_post:
            fit_obj.corner_post(samples_phys, event_name, path, truths)
        if plot_trace:
            # Assuming traceplot exists
            fit_obj.traceplot(sampler, event_name, path, truths)
        if plot_run:
            # Assuming runplot exists
            fit_obj.runplot(sampler, event_name, path)

        if plot_final_figures:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 2]},
                figsize=(12, 8),
            )
            colours = ["orange", "red", "green"]
            data_labels = ["W146", "Z087", "K213"]
            ns = 20

            # This should use samples_phys or a representation of the best fit
            # from samples. For now, using truths as a placeholder for p0.
            p0 = truths["params"][:ndim]
            # Sets the event_fit to use these parameters for A0_fit calculation
            event_fit.set_params(p0)

            fs_ref, fb_ref = {}, {}
            for obs in event_fit.data.keys():
                # OLD PROBLEMATIC LINE:
                # t_obs, f_obs, ferr_obs = event_fit.data[obs]

                # CORRECTED UNPACKING:
                current_data_array = event_fit.data[obs]
                t_obs = current_data_array[0, :]
                f_obs = current_data_array[1, :]
                ferr_obs = current_data_array[2, :]
                # END CORRECTION

                A_ref = event_fit.get_magnification(t_obs, obs)
                fs_ref[obs], fb_ref[obs] = fit_obj.get_fluxes(
                    A_ref,
                    f_obs,
                    ferr_obs**2,
                )

            tt = np.linspace(
                np.min(event_fit.data[0][0]),
                np.max(event_fit.data[0][0]),
                2000,
            )
            A0_fit = event_fit.get_magnification(tt, obs=0)
            ax1.plot(
                tt,
                A0_fit,
                "--",
                color="r",
                alpha=0.7,
                zorder=5,
                label="Truth Model",
            )

            chain_index = np.random.randint(0, len(samples_phys), ns)
            for k in range(ns):
                p = samples_phys[chain_index[k], :]
                event_fit.set_params(p)
                A_p = event_fit.get_magnification(tt, obs=0)
                ax1.plot(tt, A_p, "-", color="blue", alpha=0.1, zorder=10)

            for obs in event_fit.data.keys():
                t, f = event_fit.data[obs][0], event_fit.data[obs][1]
                A_data = (f - fb_ref[obs]) / fs_ref[obs]
                ax1.plot(
                    t,
                    A_data,
                    ".",
                    color=colours[obs],
                    label=data_labels[obs],
                    alpha=0.5,
                    zorder=1,
                )

            ax1.set_ylabel("Magnification")
            ax2.set_ylabel("Residuals")
            ax2.set_xlabel("BJD")

            if LOM_enabled:
                title_str = (
                    "s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, "
                    "\ntE=%.2f, piEE=%.2f, piEN=%.2f, "
                    "i=%.2f, phase=%.2f, period=%.2f"
                )
            else:
                title_str = (
                    "s=%.2f, q=%.6f, rho=%.6f, u0=%.2f, alpha=%.2f, t0=%.2f, "
                    "\ntE=%.2f, piEE=%.2f, piEN=%.2f"
                )
            ax1.set_title(title_str % tuple(p0))
            ax1.legend()
            plt.savefig(
                path + "posteriors/" + event_name + "_lightcurve_samples.png",
                dpi=300,
            )
            plt.close(fig)

            # --- Magnitude Lightcurve Plot ---
            mag_zp = 25

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 2]},
                figsize=(12, 8),
            )

            F = fs_ref[0] * A0_fit + fb_ref[0]
            mag = -2.5 * np.log10(F - mag_zp)
            ax1.plot(
                tt,
                mag,
                "--",
                color="r",
                alpha=0.7,
                zorder=5,
                label="Truth Model",
            )

            chain_index = np.random.randint(0, len(samples_phys), ns)
            for k in range(ns):
                p = samples_phys[chain_index[k], :]
                event_fit.set_params(p)
                A_p = event_fit.get_magnification(tt, obs=0)
                Fp = fs_ref[0] * A_p + fb_ref[0]
                mag_p = -2.5 * np.log10(Fp - mag_zp)
                ax1.plot(tt, mag_p, "-", color="blue", alpha=0.1, zorder=10)

            for obs in event_fit.data.keys():
                t = event_fit.data[obs][0]
                f = event_fit.data[obs][1]
                mag_data = -2.5 * np.log10(f - mag_zp)
                ax1.plot(
                    t,
                    mag_data,
                    ".",
                    color=colours[obs],
                    label=data_labels[obs],
                    alpha=0.5,
                    zorder=1,
                )

            ax1.set_ylabel("Magnitude")
            ax1.invert_yaxis()
            ax2.set_ylabel("Residuals")
            ax2.set_xlabel("BJD")
            ax1.set_title(title_str % tuple(p0))
            ax1.legend()
            plt.savefig(
                path + "posteriors/" + event_name + "_mag_lightcurve_samples.png",
                dpi=300,
            )
            plt.close(fig)

        end_final_figures = time.time()
        print("Time to make final figures = ", end_final_figures - end_sampler)

        # --- Event Completion ---
        print(f"Event {i} ({event_name}) is done")
        if not os.path.exists(path + "emcee_complete.txt"):
            np.savetxt(path + "emcee_complete.txt", np.array([]), fmt="%s")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            complete_list = np.atleast_1d(
                np.loadtxt(path + "emcee_complete.txt", dtype=str)
            )
            complete_list = np.hstack([complete_list, event_name])
            np.savetxt(path + "emcee_complete.txt", complete_list, fmt="%s")

    end_time = time.time()
    print("\n\n--- Timing Summary ---")
    print("Total time = ", end_time - start_time)
    print("--------------------------------\n\n\n")
