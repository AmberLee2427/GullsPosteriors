"""Data handling utilities for the microlensing challenge."""

import numpy as np
import sys
import os
import pandas as pd
import warnings


class Data:
    """Interface for loading light curves and truth parameters.

    The class provides methods to iterate through Data Challenge
    events and to parse the corresponding light curve files and master
    CSV tables.
    """

    def __init__(self):
        """Create an empty container for event data.

        Notes
        -----
        The constructor does not set any attributes or perform I/O.  Side
        effects are produced when calling :meth:`new_event` which writes the
        files ``emcee_run_list.txt`` and ``emcee_complete.txt`` in the
        specified directory.
        """

        pass

    def new_event(self, path, sort="alphanumeric"):
        r"""Return the next lightcurve and its true parameters.

        Parameters
        ----------
        path : str
            Directory containing the Data Challenge files.  The directory must
            include the master ``*.csv`` file as well as the ``*.det.lc``
            lightcurve files.
        sort : str, optional
            Selection method for the next event.  Only ``'alphanumeric'`` is
            currently supported and will process files in lexicographic order.

        Returns
        -------
        tuple
            ``(event_name, truths, data)`` where ``event_name`` is the event
            identifier, ``truths`` is a :class:`pandas.Series` with the event
            parameters and additional derived values, and ``data`` is the
            dictionary returned by :meth:`load_data`.

        Notes
        -----
        The function maintains two files in ``path``:

        ``emcee_run_list.txt``
            Records lightcurve files that have been processed.  The file will
            be created if it does not already exist and the selected file will
            be appended to it.
        ``emcee_complete.txt``
            Created if missing.  The file is not modified by this routine but
            is expected by later stages of the pipeline.

        ``new_event`` modifies ``emcee_run_list.txt`` on every successful call
        and therefore has file-system side effects.
        """

        files = os.listdir(path)
        files = sorted(files)

        if path[-1] != "/":
            path = path + "/"

        if not os.path.exists(
            path + "emcee_run_list.txt"
        ):  # if the run list doesn't exist, create it
            run_list = np.array([])
            np.savetxt(path + "emcee_run_list.txt", run_list, fmt="%s")

        if not os.path.exists(
            path + "emcee_complete.txt"
        ):  # if the complete list doesn't exist, create it
            complete_list = np.array([])
            np.savetxt(path + "emcee_complete.txt", complete_list, fmt="%s")

        for file in files:
            if "csv" in file:
                master_file = path + file

        if sort == "alphanumeric":

            for file in files:

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    run_list = np.loadtxt(
                        path + "emcee_run_list.txt", dtype=str
                    )

                if (file not in run_list) and ("det.lc" in file):

                    print("Already ran:", run_list)
                    run_list = np.hstack([run_list, file])
                    print("Running:", file, type(file))
                    np.savetxt(path + "emcee_run_list.txt", run_list, fmt="%s")

                    lc_file_name = file.split(".")[0]
                    event_identifiers = lc_file_name.split("_")
                    event_id = event_identifiers[-1]
                    sub_run = event_identifiers[
                        -3
                    ]  # the order of these is fucked up
                    field = event_identifiers[
                        -2
                    ]  # and this one. -A 2024-11-11 resample

                    data_file = path + file

                    data = self.load_data(
                        data_file
                    )  # bjd, flux, flux_err, tshift, ushift

                    event_name = f"{field}_{sub_run}_{event_id}"
                    # print('event_name = ', event_name)

                    obs0_data = data[0].copy()
                    simt = obs0_data[7]
                    bjd = obs0_data[0]

                    truths = self.get_params(
                        master_file, event_id, sub_run, field, simt, bjd
                    )
                    # turns all the degress to radians and sim time to bjd
                    break

        """if ".txt" in sort:
            files = np.loadtxt(sort)
            for i in range(len(files)):
                if os.path.exists('runlist.npy'):
                    runlist = np.loadtxt('runlist.npy')
                else:
                    runlist = np.array([])
                if files[i] not in runlist:
                    runlist = np.vstack(files[i])
                    np.savetxt('runlist.txt', runlist, fmt='%s')
                    data = mm.MulensData(file_name='data/' + files[i])
                    true_params = np.loadtxt(
                        'true_params/' + files[i].split('.')[0] + '.txt'
                    )
                    break"""

        print()
        # This is fucking dumb, but the 'lcname's in the master file do not
        # match the actual lc file names
        if (
            file == truths["lcname"]
        ):  # check that the data file and true params match
            print("Data file and true params 'lcname' match")
            sys.exit()
            # return event_name, truths, data
        else:
            print("Data file and true params 'lcname' do not match")
            print(file, "!=", truths["lcname"])
            if len(file) != len(truths["lcname"]):
                print("length:", len(file), "!=\n", len(truths["lcname"]))
            return event_name, truths, data

    def load_data(self, data_file):
        r"""Load a Data Challenge lightcurve file.

        Parameters
        ----------
        data_file : str
            Path to the whitespace separated file containing the lightcurve.

        Returns
        -------
        dict
            Keys are observatory codes and values are ``(N, 8)`` arrays for
            that observatory.

        Notes
        -----
        The lightcurve columns are:
            [0] Simulation_time
            [1] measured_relative_flux
            [2] measured_relative_flux_error
            [3] true_relative_flux
            [4] true_relative_flux_error
            [5] observatory_code
            [6] saturation_flag
            [7] best_single_lens_fit
            [8] parallax_shift_t
            [9] parallax_shift_u
            [10] BJD
            [11] source_x
            [12] source_y
            [13] lens1_x
            [14] lens1_y
            [15] lens2_x
            [16] lens2_y

        Magnitudes can be computed using:

        .. math::
            m = m_{source} + 2.5 \log f_s - 2.5 \log{F}

        where :math:`F=fs*\mu + (1-fs)` is the relative flux (in the file),
        :math:`\mu` is the magnification, and

        .. math::
            \sigma_m = 2.5/\ln{10} \sigma_F/F.

        These are listed in the header information in lines ``#fs`` and
        ``#Obssrcmag`` with order matching the observatory code order.
        The observatory codes correspond to 0=W146, 1=Z087, 2=K213

        Bugs/issues/caveats:
        The output file columns list a limb darkening parameter of Gamma=0, it
        is actually Gamma=0.36 (in all filters).
        The orbit for the Z087 observatory appears to be different to the W146
        and K213 observatories.
        Dev is working on producing the ephemerides, but for single observatory
        parallax, using interpolated versions of the ones available for the
        data challenge will probably be accurate enough, or an Earth ephemeris
        with
        the semimajor axis (but not period) increased by 0.01 AU.
        Lenses with masses smaller than the isochrone grid limits (I believe
        0.1 MSun) will have filler values for magnitudes and lens stellar
        properties.
        There may be some spurious detections in the list where the single lens
        fit failed. Please let dev know if you find any of these events so that
        we can improve the single lens fitter."""

        header = [
            "Simulation_time",
            "measured_relative_flux",
            "measured_relative_flux_error",
            "true_relative_flux",
            "true_relative_flux_error",
            "observatory_code",
            "saturation_flag",
            "best_single_lens_fit",
            "parallax_shift_t",
            "parallax_shift_u",
            "BJD",
            "source_x",
            "source_y",
            "lens1_x",
            "lens1_y",
            "lens2_x",
            "lens2_y",
            "A",
            "B",
            "C",
        ]

        data = pd.read_csv(
            data_file, sep=r"\s+", skiprows=12, names=header
        )  # delim_whitespace=True is the same as sep=r'\s+', but older.
        # The 'r' in sep=r'\s+' means raw string, which is not necessary.
        # Otherwise you get annoying warnings.

        # simulation time to BJD
        print(data["BJD"][0])
        print(data["Simulation_time"][0])

        self.sim_time0 = np.sum(data["BJD"] - data["Simulation_time"]) / len(
            data["BJD"]
        )

        data = data[
            [
                "BJD",
                "measured_relative_flux",
                "measured_relative_flux_error",
                "parallax_shift_t",
                "parallax_shift_u",
                "observatory_code",
                "true_relative_flux",
                "true_relative_flux_error",
                "Simulation_time",
            ]
        ]

        data_dict = {}
        for code in data["observatory_code"].unique():
            data_obs = data[data["observatory_code"] == code][
                [
                    "BJD",
                    "measured_relative_flux",
                    "measured_relative_flux_error",
                    "parallax_shift_t",
                    "parallax_shift_u",
                    "true_relative_flux",
                    "true_relative_flux_error",
                    "Simulation_time",
                ]
            ].reset_index(drop=True)
            data_dict[code] = data_obs.to_numpy().T

        return data_dict

    def get_params(
        self, master_file, event_id, sub_run, field, epoch=None, bjd=None
    ):
        r"""Return the true parameters for an event.

        Parameters
        ----------
        master_file : str
            Path to the master ``*.csv`` file containing event information.
        event_id : int or str
            Identifier of the event in ``master_file``.
        sub_run : int or str
            Data Challenge sub-run number.
        field : int or str
            Field identifier within the Data Challenge.
        epoch : array-like, optional
            Simulation epochs used to convert times to BJD.
        bjd : array-like, optional
            Barycentric Julian Date corresponding to ``epoch``.

        Returns
        -------
        pandas.Series
            Series of event parameters with additional keys ``params`` and
            ``tcroin`` that have been converted to BJD.

        Notes
        -----
        Several quantities are converted for convenience:

        * ``alpha``, ``Planet_inclination`` and ``Planet_orbphase`` are
          converted from degrees to radians.
        * ``t0lens1`` and ``tcroin`` are converted from simulation time to BJD
          either by interpolation of ``epoch``/``bjd`` or by using the
          offset determined from the lightcurve.
        * ``Planet_period`` is converted from years to days.
        """
        event_id = int(event_id)
        sub_run = int(sub_run)
        field = int(field)

        master = pd.read_csv(master_file, header=0, delimiter=",")
        # print(master.head())

        truths = master[
            (master["EventID"] == int(event_id))
            & (master["SubRun"] == int(sub_run))
            & (master["Field"] == int(field))
        ].iloc[0]

        # print(self.sim_time0)

        s = truths["Planet_s"]
        q = truths["Planet_q"]
        rho = truths["rho"]
        u0 = truths["u0lens1"]  # croin
        alpha = truths["alpha"] * np.pi / 180  # convert to radians
        if epoch is not None and bjd is not None:
            t0_sim = truths["t0lens1"]
            t0_bjd = np.interp(t0_sim, epoch, bjd)
            t0 = t0_bjd
            tc_sim = truths["tcroin"]
            tc_bjd = np.interp(tc_sim, epoch, bjd)
            tcroin = tc_bjd
        else:
            t0 = truths["t0lens1"] + self.sim_time0  # convert to BJD
            tcroin = truths["tcroin"] + self.sim_time0  # convert to BJDs
        truths["t0lens1"] = t0
        tE = truths["tE_ref"]
        piEE = truths["piEE"]
        piEN = truths["piEN"]
        i = truths["Planet_inclination"] * np.pi / 180  # convert to radians
        phase = (
            truths["Planet_orbphase"] * np.pi / 180
        )  # convert to radians # centre on tcroin
        period = truths["Planet_period"] * 365.25  # convert to days
        # phase_change = truths['tcroin'] / period
        # phase = phase + phase_change  # centre on t0
        # phase = phase % (2.0*np.pi)  # make sure it's between 0 and 2pi
        truths["params"] = [
            s,
            q,
            rho,
            u0,
            alpha,
            t0,
            tE,
            piEE,
            piEN,
            i,
            phase,
            period,
        ]

        truths["tcroin"] = tcroin

        return truths
