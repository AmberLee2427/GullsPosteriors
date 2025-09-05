"""Fitting utilities wrapping emcee and dynesty samplers."""

# In Fit/__init__.py

import sys
import numpy as np


class Fit:
    """Wrapper for fitting microlensing models.

    The class interfaces with either the :mod:`emcee` ensemble sampler or the
    :mod:`dynesty` nested sampler to explore parameter space.  In addition to
    providing log-likelihood and prior calculations, it exposes a number of
    helper routines for running the samplers and generating diagnostic plots.

    Parameters
    ----------
    sampling_package : {'emcee', 'dynesty'}, optional
        Backend used for sampling.
    debug : list of str or None, optional
        Substrings enabling verbose debug output.
    LOM_enabled : bool, optional
        If ``True``, include lens orbital motion parameters in the model.
    ndim : int or None, optional
        Number of free parameters.
    labels : list of str or None, optional
        Parameter labels used when creating plots.

    Attributes
    ----------
    sampling_package : str
        Name of the sampling backend currently in use.
    debug : list of str
        Keywords that activate additional console output.
    LOM_enabled : bool
        Flag indicating whether orbital motion parameters are fitted.
    ndim : int or None
        Dimensionality of the model.
    labels : list of str or None
        Labels corresponding to each model parameter.
    """

    from ._emcee import (
        run_emcee,
        run_burnin,
        lnprob_transform,
        plot_chain,
        corner_post,
    )

    # prior_transform will now be fully defined in _dynesty.py
    # runplot and traceplot are also in _dynesty.py
    from ._dynesty import prior_transform, runplot, traceplot, detransform_theta, run_dynesty

    # MODIFIED __init__ to accept and store ndim and labels
    def __init__(
        self,
        sampling_package="emcee",
        debug=None,
        LOM_enabled=True,
        ndim=None,
        labels=None,
        show_progress=False,
        sigma_fb=50.0,
        sigma_rho=0.1,
        sigma_q=0.1,
        sigma_s=10.0,
        normal=True,
        unit_cube=False
    ):
        """Initialise a sampler wrapper.

        Parameters
        ----------
        sampling_package : {'emcee', 'dynesty'}, optional
            Backend used to perform the sampling.
        debug : list of str or None, optional
            Strings enabling additional console output.
        LOM_enabled : bool, optional
            Include lens orbital motion parameters when ``True``.
        ndim : int or None, optional
            Number of parameters in the model.
        labels : list of str or None, optional
            Parameter labels used for corner plots.
        show_progress : bool, optional
            Whether to show progress bars during MCMC sampling.
        sigma_fb : float, optional
            Standard deviation for the Gaussian prior on negative blend flux.

        Attributes
        ----------
        sampling_package : str
            Name of the chosen sampling backend.
        debug : list of str
            Debug keywords stored from ``debug``.
        LOM_enabled : bool
            Flag controlling orbital motion physics.
        ndim : int or None
            Dimensionality of the parameter space.
        labels : list of str or None
            Labels for each parameter.
        show_progress : bool
            Whether to show progress bars during MCMC sampling.
        current_event : Event or None
            The current microlensing event being fitted.
        sigma_fb : float
            Standard deviation for the Gaussian prior on negative blend flux.
        sigma_rho : float
            Standard deviation for the Gaussian prior on rho.
        sigma_q : float
            Standard deviation for the Gaussian prior on q.
        sigma_s : float
            Standard deviation for the Gaussian prior on s.
        normal : bool
            If ``True``, use normal priors (default).
        unit_cube : bool
            If ``True``, sample in the unit cube (for dynesty).
        """
        if debug is not None:
            self.debug = debug
        else:
            self.debug = []

        self.sampling_package = sampling_package
        if sampling_package != "dynesty" and sampling_package != "emcee":
            print("Invalid sampling package. Must be dynesty or emcee")
            sys.exit()

        self.LOM_enabled = LOM_enabled
        self.ndim = ndim  # NEW: Store ndim
        self.labels = labels  # NEW: Store labels
        self.show_progress = show_progress  # NEW: Store show_progress
        self.current_event = None  # NEW: Store current event
        self.sigma_fb = sigma_fb  # NEW: Store sigma_fb
        self.sigma_rho = sigma_rho  # NEW: Store sigma_rho
        self.sigma_q = sigma_q  # NEW: Store sigma_q
        self.sigma_s = sigma_s  # NEW: Store sigma_s
        self.normal = normal  # NEW: Store normal
        self.unit_cube = unit_cube  # NEW: Store unit_cube

    def get_fluxes(self, model: np.ndarray, f: np.ndarray, sig2: np.ndarray):
        """Solve for the source and blend fluxes.

        Parameters
        ----------
        model : numpy.ndarray
            Model magnification curve.
        f : numpy.ndarray
            Observed flux measurements.
        sig2 : numpy.ndarray
            Variance of the observed fluxes.

        Returns
        -------
        FS : float
            Best-fit source flux.
        FB : float
            Best-fit blend flux.
        """
        if model.shape[0] != f.shape[0]:
            print("debug Fit.get_fluxes: model and f have different lengths")
            sys.exit()
        if model.shape[0] != sig2.shape[0]:
            print(
                "debug Fit.get_fluxes: model and sig2 have different lengths"
            )
            sys.exit()
        if f.shape[0] != sig2.shape[0]:
            print("debug Fit.get_fluxes: f and sig2 have different lengths")
            sys.exit()

        # Check for pathological magnification values
        if np.all(model == 0) or np.any(~np.isfinite(model)):
            raise ValueError(f"Invalid magnification model: all zeros or contains non-finite values. Model range: [{np.min(model)}, {np.max(model)}]")

        # A
        A11 = np.sum(model**2 / sig2)
        Adiag = np.sum(model / sig2)
        A22 = np.sum(1.0 / sig2)
        A = np.array([[A11, Adiag], [Adiag, A22]])

        # Check for singular matrix before solving
        det_A = A[0,0] * A[1,1] - A[0,1] * A[1,0]
        if abs(det_A) < 1e-15:
            raise ValueError(f"Singular flux fitting matrix. Determinant: {det_A}, A11: {A11}, A22: {A22}, Adiag: {Adiag}")

        # C
        C1 = np.sum((f * model) / sig2)
        C2 = np.sum(f / sig2)
        C = np.array([C1, C2]).T

        # B
        B = np.linalg.solve(A, C)
        FS = float(B[0])
        FB = float(B[1])

        if "fluxes" in self.debug:
            print("debug Fit.get_fluxes: A: ", A)
            print("debug Fit.get_fluxes: C: ", C)
            print("debug Fit.get_fluxes: B: ", B)
            print("debug Fit.get_fluxes: FS: ", FS)
            print("debug Fit.get_fluxes: FB: ", FB)

        return FS, FB

    def get_chi2(self, event, params, measured_flux=True):
        """Compute the chi-square values for a given parameter set.

        Parameters
        ----------
        event : Event
            Microlensing event object containing the light curve data and
            magnification model.
        params : array_like
            Parameter values to apply via ``event.set_params``.

        Returns
        -------
        dict
            Dictionary mapping each observatory to its chi-square array.
        float
            Total chi-square summed over all observatories.

        Notes
        -----
        The method updates ``event`` with ``params`` and then, for each
        observatory, derives the best-fit source and blend fluxes.  These
        fluxes are used to compute the chi-square contribution of that
        observatory which is then accumulated into the total.
        """
        if "chi2" in self.debug:
            print("debug Fit.get_chi2: params: ", params)
            print("debug Fit.get_chi2: event type: ", type(event))

        event.set_params(params)
        chi2sum = 0.0
        chi2 = {}

        for obs in event.data.keys():  # looping through observatories
            t = event.data[obs][0]  # BJD
            if measured_flux:
                f = event.data[obs][1]  # obs_rel_flux
                f_err = event.data[obs][2]  # obs_rel_flux_err
            else:
                f = event.data[obs][5]  # true_rel_flux
                f_err = event.data[obs][6]  # true_rel_flux_err

            A = event.get_magnification(t, obs)
            fs, fb = self.get_fluxes(A, f, f_err**2)

            chi2[obs] = ((f - (A * fs + fb)) / f_err) ** 2

            chi2sum += np.sum(chi2[obs])

            if "chi2" in self.debug:
                print("debug Fit.get_chi2: obs: ", obs)
                print("debug Fit.get_chi2: t: ", t)
                print("debug Fit.get_chi2: f: ", f)
                print("debug Fit.get_chi2: f_err: ", f_err)
                print("debug Fit.get_chi2: A: ", A)
                print("debug Fit.get_chi2: fs: ", fs)
                print("debug Fit.get_chi2: fb: ", fb)
                print("debug Fit.get_chi2: chi2: ", chi2[obs])
                print("debug Fit.get_chi2: chi2sum: ", chi2sum)

        return chi2, chi2sum

    def lnlike(self, theta, event):
        """Compute the log-likelihood for ``theta``.

        Parameters
        ----------
        theta : array_like
            Parameter vector to evaluate.
        event : Event
            Microlensing event containing data and model information.

        Returns
        -------
        float
            The log-likelihood value.
        """
        _, chi2 = self.get_chi2(event, theta)

        if "lnlike" in self.debug:
            print("debug Fit.lnlike: chi2: ", chi2)
            print("debug Fit.lnlike: theta: ", theta)

        return -0.5 * chi2

    # MODIFIED: lnprior to use self.LOM_enabled
    def lnprior(self, theta, event=None):
        """Evaluate the log-prior probability.

        Parameters
        ----------
        theta : array_like
            Parameter vector to evaluate.
        event : Event or None, optional
            Microlensing event containing data and model information.

        Returns
        -------
        float
            Log-prior probability or ``-np.inf`` if outside bounds.
        """
        if self.current_event is not None:
            current_event = self.current_event
        if event is not None:  # If event is provided use that as priority
            current_event = event
            
        if self.LOM_enabled:
            s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period = theta
            if "ln_prior" in self.debug:
                print("debug Fit.lnprior (LOM):", theta)
            if (
                tE > 0.0
                and q <= 1.0
                and period / tE > 4
                and s > 0.001
                and rho > 0.0
            ):
                    
                # Only check blend flux if we have a current event
                if current_event is not None:
                    # Get blend flux for this parameter set
                    t = current_event.data[list(current_event.data.keys())[0]][0]  # Get times from first observatory
                    A = current_event.get_magnification(t, list(current_event.data.keys())[0])
                    f = current_event.data[list(current_event.data.keys())[0]][1]  # Get fluxes
                    f_err = current_event.data[list(current_event.data.keys())[0]][2]  # Get errors
                    _, fb = self.get_fluxes(A, f, f_err**2)
                    
                    # Add Gaussian prior on negative blend flux
                    if self.normal and not self.unit_cube:
                        if fb < 0:
                            # Allow small negative values but penalize large ones
                            return -0.5 * (fb / self.sigma_fb)**2
                        if q > 1:  # gently disuade primary swapping
                            return -0.5 * ((q - 1) / self.sigma_q)**2
                        if rho > 1:  # gently disuade unphysically large sources
                            return -0.5 * ((rho - 1) / self.sigma_rho)**2
                        if s > 10:  # gently disuade very wide binaries
                            return -0.5 * ((s - 10) / self.sigma_s)**2
                return 0.0
            else:
                return -np.inf
        else:  # No LOM
            s, q, rho, u0, alpha, t0, tE, piEE, piEN = theta
            if "ln_prior" in self.debug:
                print("debug Fit.lnprior (No LOM):", theta)
            if tE > 0.0 and q <= 1.0 and q > 0.0 and s > 0.001 and rho > 0.0:
                # Only check blend flux if we have a current event
                if current_event is not None:
                    # Get blend flux for this parameter set
                    t = current_event.data[list(current_event.data.keys())[0]][0]  # Get times from first observatory
                    A = current_event.get_magnification(t, list(current_event.data.keys())[0])
                    f = current_event.data[list(current_event.data.keys())[0]][1]  # Get fluxes
                    f_err = current_event.data[list(current_event.data.keys())[0]][2]  # Get errors
                    _, fb = self.get_fluxes(A, f, f_err**2)
                    
                    # Add Gaussian prior on negative blend flux
                    if self.normal and not self.unit_cube:
                        if fb < 0:
                            # Allow small negative values but penalize large ones
                            return -0.5 * (fb / self.sigma_fb)**2
                        if q > 1:  # gently disuade primary swapping
                            return -0.5 * ((q - 1) / self.sigma_q)**2
                        if rho > 1:  # gently disuade unphysically large sources
                            return -0.5 * ((rho - 1) / self.sigma_rho)**2
                        if s > 10:  # gently disuade very wide binaries
                            return -0.5 * ((s - 10) / self.sigma_s)**2
                return 0.0
            else:
                return -np.inf

    # MODIFIED: lnprob to use self.LOM_enabled
    def lnprob(self, theta, event):
        """Calculate the log-posterior probability.

        Parameters
        ----------
        theta : array_like
            Parameter vector to evaluate.
        event : Event
            Microlensing event providing data and magnification model.

        Returns
        -------
        float
            Sum of log-prior and log-likelihood values.
        """
        # make a copy of theta, so that it doesn't get inadvertedly edited in the case where it is a list
        params = theta.copy()
        
        # Define which parameter names are log-transformed
        log_param_names_base = ["s", "q", "rho", "tE"]

        # Complete the log_param_names logic
        if self.LOM_enabled:
            log_param_names = log_param_names_base + ["period"]
        else:
            log_param_names = log_param_names_base

        # Apply log transformation to the relevant parameters
        for i, name in enumerate(log_param_names):
            if name in log_param_names:
                params[i] = 10**(params[i])

        if self.LOM_enabled:
            params[4] %= 2 * np.pi  # alpha
            params[10] %= 2 * np.pi  # phase
            # For inclination 'i', it's usually 0 to pi. If it's 0 to 2pi in
            # your setup, this is fine.
            # Otherwise, you might need params[9] = np.abs(params[9] % np.pi) or
            # similar.
            params[9] %= 2 * np.pi  # i

        else:
            params[4] %= 2 * np.pi  # alpha

        lp = self.lnprior(params, event)
        if not np.isfinite(lp):
            return -np.inf

        ll = self.lnlike(params, event)
        if not np.isfinite(ll):
            return -np.inf

        if "lnprob" in self.debug:
            print("debug Fit.lnprob: lp, ll: ", lp, ll)
            print("                  ", params)

        return lp + ll
