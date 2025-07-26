import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp

def run_emcee(
    self,
    nl,
    ndim,
    stepi,
    mi,
    log_prob_function,
    state,
    event_obj,
    truths, # This is the truths dictionary
    prange_linear,
    prange_log,
    normal,
    threads=1,
    event_name="",
    path="./",
    labels=None,
    fisher_uncertainties_for_prior=None, # Added this argument
    fisher_uncertainties_for_plotting=None # Added this argument
):
    """Run an ``emcee`` ensemble sampler.

    Parameters
    ----------
    nl : int
        Number of walkers.
    ndim : int
        Number of parameters in the model.
    stepi : int
        Steps to take between checkpoints.
    mi : int
        Total number of sampling steps.
    log_prob_function : callable
        Function returning the log-probability.
    state : array_like
        Initial state of the walkers.
    event_obj : Event
        Microlensing event used to compute the likelihood.
    truths : dict
        Dictionary of reference parameter values.
    prange_linear : array_like
        Width of uniform priors for linear parameters.
    prange_log : array_like
        Width (in dex) of priors for log parameters.
    normal : bool
        If ``True``, draw from normal rather than uniform priors.
    threads : int, optional
        Number of worker processes used by ``emcee``.
    event_name : str, optional
        Prefix for saved diagnostic files.
    path : str, optional
        Directory in which output files are written.
    labels : list of str or None, optional
        Parameter labels for plotting.
    fisher_uncertainties_for_prior : array_like or None, optional
        1-sigma Fisher uncertainties used to define the prior ranges.
    fisher_uncertainties_for_plotting : array_like or None, optional
        1-sigma Fisher uncertainties used for plotting.

    Returns
    -------
    emcee.EnsembleSampler
        The sampler instance after completion.
    """
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    # Set the current event for the prior
    self.current_event = event_obj

    # The new arguments to be passed to the log probability function
    # truths (the dictionary) is passed here, and lnprob_transform will extract truths['params']
    # Pass fisher_uncertainties_for_prior to log_prob_function
    log_prob_args = [event_obj, truths, prange_linear, prange_log, normal, fisher_uncertainties_for_prior]

    if threads > 1:
        with Pool(threads) as pool:
            # Initialize the sampler with the new arguments
            sampler = emcee.EnsembleSampler(
                nl, ndim, log_prob_function, args=log_prob_args, pool=pool
            )

            # --- The rest of the function is identical ---
            # Run the sampler
            count = 0
            steps = 0
            while steps < mi:
                state, lnp, _ = sampler.run_mcmc(state, stepi, progress=self.show_progress)
                flatchain = sampler.flatchain
                flatlnprobability = sampler.flatlnprobability

                # Save the samples
                np.save(
                    path + "posteriors/" + event_name + "_emcee_samples.npy",
                    flatchain,
                )
                np.save(
                    path + "posteriors/" + event_name + "_emcee_lnprob.npy",
                    flatlnprobability,
                )
                np.save(
                    path + "posteriors/" + event_name + "_emcee_state.npy",
                    state,
                )

                # Pass the actual truths['params'] and fisher_uncertainties to plot_chain
                # Also pass prange_linear, prange_log, and normal for detransform_theta
                self.plot_chain(sampler, event_name, path, labels=labels,
                                truths=truths['params'], # Pass the actual physical truths array
                                fisher_uncertainties_for_plotting=fisher_uncertainties_for_plotting,
                                fisher_uncertainties_for_prior=fisher_uncertainties_for_prior,
                                prange_linear=prange_linear,
                                prange_log=prange_log,
                                normal=normal)

                steps += stepi
                count += 1

    else:
        # Initialize the sampler with the new arguments
        sampler = emcee.EnsembleSampler(
            nl, ndim, log_prob_function, args=log_prob_args
        )

        # --- The rest of the function is identical ---
        # Run the sampler
        steps = 0
        count = 0
        while steps < mi:
            state, lnp, _ = sampler.run_mcmc(state, stepi, progress=self.show_progress)
            flatchain = sampler.flatchain
            flatlnprobability = sampler.flatlnprobability

            # Save the samples
            np.save(
                path + "posteriors/" + event_name + "_emcee_samples.npy",
                flatchain,
            )
            np.save(
                path + "posteriors/" + event_name + "_emcee_lnprob.npy",
                flatlnprobability,
            )
            np.save(
                path + "posteriors/" + event_name + "_emcee_state.npy", state
            )

            self.plot_chain(sampler, event_name, path, labels=labels,
                            truths=truths['params'], # Pass the actual physical truths array
                            fisher_uncertainties_for_plotting=fisher_uncertainties_for_plotting,
                            fisher_uncertainties_for_prior=fisher_uncertainties_for_prior,
                            prange_linear=prange_linear,
                            prange_log=prange_log,
                            normal=normal)

            steps += stepi
            count += 1

    return sampler


def run_burnin(
    self,
    nl,
    ndim,
    stepi,
    log_prob_function,
    state,
    event_obj,
    truths, # This is the truths dictionary
    prange_linear,
    prange_log,
    p_unc, # This is your adaptive prior width array
    normal,
    max_steps=1000,
    threads=1,
    event_name="",
    path="./",
    labels=None,
    min_steps=500,
    fisher_uncertainties_for_plotting=None # Added this argument
):
    """Run a short ``emcee`` burn-in phase expanding priors as needed.

    After every ``stepi`` iterations the walker coordinates are inspected. If a
    significant fraction of walkers crowd the unit-cube boundaries the prior
    widths are increased by 20\% and the walker coordinates are rescaled to the
    enlarged cube. The routine stops when ``max_steps`` is reached or when two
    consecutive iterations require no further expansion.

    Parameters
    ----------
    nl : int
        Number of walkers.
    ndim : int
        Number of model parameters.
    stepi : int
        Interval between sampler checkpoints.
    log_prob_function : callable
        Function computing the log-probability.
    state : :class:`emcee.State`
        Initial state of the walkers.
    event_obj : Event
        Microlensing event providing the likelihood.
    truths : array_like
        Central values defining the priors.
    prange_linear : array_like
        Half-widths for linear parameters.
    prange_log : array_like
        Half-widths in dex for log parameters.
    p_unc : array_like
        Array of parameter uncertainties to update.
    normal : bool
        If ``True``, use normal rather than uniform priors.
    max_steps : int, optional
        Maximum number of burn-in steps to run.
    threads : int, optional
        Number of worker processes.
    event_name : str, optional
        Prefix for saved diagnostic files.
    path : str, optional
        Directory where output is written.
    labels : list of str or None, optional
        Parameter labels.
    min_steps : int, optional
        Minimum number of steps to run.
    fisher_uncertainties_for_plotting : array_like or None, optional
        1-sigma Fisher uncertainties used for plotting only. 
        Fisher priors and adaptive widths are not currently supported.

    Returns
    -------
    emcee.State
        Final sampler state after burn-in.
    ndarray
        Updated ``p_unc`` array reflecting any prior expansions.
    """

    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    # Set the current event for the prior
    self.current_event = event_obj

    # truths (the dictionary) is passed here, and lnprob_transform will extract truths['params']
    # Pass fisher_uncertainties_for_prior to log_prob_function
    log_prob_args = [event_obj, truths, prange_linear, prange_log, normal]

    labels = labels if labels is not None else self.labels

    if threads > 1:
        pool_ctx = Pool(threads)
    else:
        pool_ctx = None

    sampler = emcee.EnsembleSampler(
        nl, ndim, log_prob_function, args=log_prob_args, pool=pool_ctx
    )

    log_param_names = ["s", "q", "rho"]
    if self.LOM_enabled:
        log_param_names.append("period")
    log_indices = [i for i, l in enumerate(labels) if l in log_param_names]
    lin_indices = [i for i in range(ndim) if i not in log_indices]

    steps = 0
    no_expand = 0
    expansion_threshold = 0.2  # Lower threshold to be more sensitive
    expansion_rate = 1.1  # More gradual expansion

    while steps < max_steps and (no_expand < 2 or steps < min_steps):
        state, _, _ = sampler.run_mcmc(state, stepi, progress=self.show_progress)

        np.save(path + f"posteriors/{event_name}_burnin_samples.npy", sampler.flatchain)
        np.save(path + f"posteriors/{event_name}_burnin_lnprob.npy", sampler.flatlnprobability)
        np.save(path + f"posteriors/{event_name}_burnin_state.npy", state)

        print(f"Fisher uncertainties: {fisher_uncertainties_for_plotting}")

        # Plotting burn-in chain - pass the physical truths and other prior params
        self.plot_chain(sampler, f"{event_name}_burnin", path, labels=labels,
                        truths=truths['params'], # Pass the actual physical truths array
                        fisher_uncertainties_for_plotting=fisher_uncertainties_for_plotting,
                        prange_linear=prange_linear,
                        prange_log=prange_log,
                        normal=normal,
                        burnin_or_post="burnin")

        # Update for emcee 3.x API
        positions = state # Access coordinates from the State object

        # Check if we need to expand the priors
        expanded = False

        # IMPORTANT: The positions here are in the *unit cube* space (0 to 1)
        # The prange_log and prange_linear are *prior widths* in physical space.
        # The expansion logic here is for adapting the *prior ranges*
        # (prange_log, prange_linear), not the p_unc (Fisher uncertainties).
        # p_unc is updated proportionally to reflect the new width.

        for j, idx in enumerate(log_indices):
            # Check for crowding at unit-cube boundaries
            frac = np.mean((positions[:, idx] < 0.05) | (positions[:, idx] > 0.95))
            if frac > expansion_threshold:
                old_range = prange_log[j]
                prange_log[j] *= expansion_rate
                # Assuming p_unc[idx] is the current 1-sigma uncertainty associated
                # with this adapted prior. If it's a fixed Fisher uncertainty,
                # it should NOT be scaled here.
                # If p_unc is *also* an adaptive width, then scaling it is fine.
                p_unc[idx] *= expansion_rate # This scales the p_unc
                new_range = prange_log[j]

                # Rescale walker positions to the new, enlarged unit cube
                # First, transform current unit-cube position to physical space using OLD range
                center = truths['params'][idx] # Use truths['params'] for the center
                old_min_log = np.log10(center) - old_range / 2.0
                phys_log = old_min_log + positions[:, idx] * old_range
                
                # Then, transform physical position to new unit-cube using NEW range
                new_min_log = np.log10(center) - new_range / 2.0
                positions[:, idx] = (phys_log - new_min_log) / new_range
                expanded = True

        for j, idx in enumerate(lin_indices):
            # Check for crowding at unit-cube boundaries
            frac = np.mean((positions[:, idx] < 0.05) | (positions[:, idx] > 0.95))
            if frac > expansion_threshold:
                old_range = prange_linear[j]
                prange_linear[j] *= expansion_rate
                # Assuming p_unc[idx] is the current 1-sigma uncertainty associated
                # with this adapted prior.
                p_unc[idx] *= expansion_rate # This scales the p_unc
                new_range = prange_linear[j]

                # Rescale walker positions to the new, enlarged unit cube
                # First, transform current unit-cube position to physical space using OLD range
                center = truths['params'][idx] # Use truths['params'] for the center
                old_min_linear = center - old_range / 2.0
                phys_linear = old_min_linear + positions[:, idx] * old_range
                
                # Then, transform physical position to new unit-cube using NEW range
                new_min_linear = center - new_range / 2.0
                positions[:, idx] = (phys_linear - new_min_linear) / new_range
                expanded = True

        if expanded:
            # Recompute log_prob for the new positions in the expanded unit cube
            # The log_prob_function takes (u, event, true, prange_linear, prange_log, normal, fisher_uncertainties)
            # where `true` is the truths dictionary.
            log_prob_values = np.array([log_prob_function(p, *log_prob_args[1:]) for p in positions])
            
            state = emcee.State(positions, log_prob_values) # Update the state object for emcee
            no_expand = 0
        else:
            no_expand += 1

        steps += stepi

    if pool_ctx is not None:
        pool_ctx.close()

    return state, p_unc, prange_linear, prange_log


def lnprob_transform(
    self, u, event, truths_dict, prange_linear, prange_log, normal=False, fisher_uncertainties_for_prior=None
):
    """Convert unit-cube samples to log-probability values.

    Parameters
    ----------
    u : array_like
        Sample from the unit hypercube.
    event : Event
        Microlensing event used to compute the likelihood.
    truths_dict : dict
        Dictionary of reference parameter values defining the prior centres.
    prange_linear : array_like
        Linear prior widths.
    prange_log : array_like
        Logarithmic prior widths in dex.
    normal : bool, optional
        If ``True``, sample the priors using normal distributions.
    fisher_uncertainties_for_prior : array_like or None, optional
        1-sigma Fisher uncertainties for each parameter.

    Returns
    -------
    float
        Log-probability of the transformed sample.
    """
    # Check if any walker is somehow outside the unit cube.
    # This check is usually handled by dynesty/emcee internally,
    # but a safeguard can be useful.
    if np.any(u < 0.0) or np.any(u > 1.0):
        return -np.inf

    # Call the new, improved prior_transform with the correct arguments
    # Pass truths_dict['params'] (the array) to prior_transform
    theta = self.prior_transform(u, truths_dict['params'], prange_linear, prange_log, normal, fisher_uncertainties_for_prior)

    # Calculate the log probability (likelihood) with the transformed
    # parameters
    lp = self.lnprob(theta, event)

    return lp


def plot_chain(self, res, event_name, path, burnin_or_post="post", labels=None, truths=None, fisher_uncertainties_for_plotting=None, fisher_uncertainties_for_prior=None, prange_linear=None, prange_log=None, normal=False):
    """Plot the trace of the walkers, with Fisher uncertainty bands and truth lines.

    Parameters
    ----------
    res : emcee.EnsembleSampler
        Sampler containing the chain and log-probability.
    event_name : str
        Name used when saving the figure.
    path : str
        Directory where the image will be written.
    burnin_or_post : str, optional
        Where the provided chains are from (burn-in or posterior collection).
    labels : list of str or None, optional
        Parameter labels for each dimension.
    truths : array_like or None, optional
        Truth values for each parameter (in physical space).
    fisher_uncertainties_for_plotting : array_like or None, optional
        1-sigma Fisher uncertainties for each parameter (in physical space).
        These do not inform the prior widths, but are used to plot the uncertainty bands.
        These are assumed to be fixed theoretical values, not adaptive prior widths.
    fisher_uncertainties_for_prior : array_like or None, optional
        1-sigma Fisher uncertainties for each parameter (in physical space).
        These are used to inform the prior widths.
        These are assumed to be fixed theoretical values, not adaptive prior widths.
    prange_linear : array_like
        Linear prior widths (needed for detransform_theta).
    prange_log : array_like
        Logarithmic prior widths in dex (needed for detransform_theta).
    normal : bool
        If ``True``, normal priors were used (needed for detransform_theta).

    Returns
    -------
    None
    """
    chain = res.chain  # shape = (nwalkers, nsteps, ndim) - This is in unit cube space
    lnprobability = res.lnprobability  # shape = (nwalkers, nsteps)
    ndim = chain.shape[2]
    nsteps = chain.shape[1]
    nwalkers = chain.shape[0]

    # detransformed truth (in unit cube space)
    # should return an ndim-D array of 0.5 values
    if truths is not None:
        if self.LOM_enabled:
            truths_for_detransform = truths[:12]  # this is brittle and should be fixed
        else:
            truths_for_detransform = truths[:9]
        # Detransform the truth values to unit cube space
        u_truths_mapped = self.detransform_theta(
            truths_for_detransform, 
            truths, 
            prange_linear,
            prange_log,
            normal,
            fisher_uncertainties_for_prior # USE THE CORRECT PRIOR DEFINITION
        )
        # do a sanity check that u_truths_mapped is an ndim-D array of 0.5 values
        if u_truths_mapped.shape[0] != ndim:
            print(f"u_truths_mapped.shape: {u_truths_mapped.shape}")
            print(f"ndim: {ndim}")
            print(f'u_truths_mapped type: {type(u_truths_mapped)}')
            raise ValueError("u_truths_mapped is not an ndim-D array")
        if not np.all(u_truths_mapped == 0.5):
            print("WARNING: u_truths_mapped is not an array of 0.5 values")
            print(f"u_truths_mapped: {u_truths_mapped}")

    if fisher_uncertainties_for_plotting is not None:
        # Create temporary arrays for truth +/- sigma for detransformation    
        theta_plus_sigma = np.array(truths_for_detransform)  # Start with full truths array
        theta_minus_sigma = np.array(truths_for_detransform)
        # Ignore the flux parametrs and (conditionally) LOM parameters
        if self.LOM_enabled: 
            theta_plus_sigma += np.array(fisher_uncertainties_for_plotting)[:12]
            theta_minus_sigma -= np.array(fisher_uncertainties_for_plotting)[:12]
        else:
            theta_plus_sigma += np.array(fisher_uncertainties_for_plotting)[:9]
            theta_minus_sigma -= np.array(fisher_uncertainties_for_plotting)[:9]

        # Detransform these values to unit cube space
        u_plus_sigma_mapped = self.detransform_theta(
            theta_plus_sigma,
            truths, # truths_array
            prange_linear,
            prange_log,
            normal,
            fisher_uncertainties_for_prior  # None if not using
        )
        
        u_minus_sigma_mapped = self.detransform_theta(
            theta_minus_sigma,
            truths, # truths_array
            prange_linear,
            prange_log,
            normal,
            fisher_uncertainties_for_prior
        )

        print(f"u_plus_sigma_mapped: {u_plus_sigma_mapped}")
        print(f"u_minus_sigma_mapped: {u_minus_sigma_mapped}")
        print(f"theta_plus_sigma: {theta_plus_sigma}")
        print(f"theta_minus_sigma: {theta_minus_sigma}")

    if labels is None:
        labels = [f"theta[{i}]" for i in range(ndim)]

    fig, axes = plt.subplots(ndim + 1, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        for j in range(nwalkers):
            ax.plot(chain[j, :, i], "k", alpha=0.1)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(labels[i])
        ax.set_ylim(-0.1, 1.1) # Expanded Y-axis limits

        # Add horizontal line for the truth (in unit cube space)
        ax.axhline(u_truths_mapped[i], color="blue", linestyle="-", linewidth=1.5, alpha=0.7, label="Truth (Unit Cube)")

        # Add horizontal dashed lines for 1-sigma uncertainties (in unit cube space)
        if fisher_uncertainties_for_plotting is not None or fisher_uncertainties_for_prior is not None:
            ax.axhline(u_plus_sigma_mapped[i], color="blue", linestyle="--", linewidth=1.0, alpha=0.7, label="Truth $\pm 1\sigma_{Fisher}$ (Unit Cube)")
            ax.axhline(u_minus_sigma_mapped[i], color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
            
        # Add legend to the first subplot only to avoid clutter
        if i == 0:
            ax.legend(loc='best')

    ax = axes[-1]
    for j in range(nwalkers):
        ax.plot(lnprobability[j], "r", alpha=0.3)
    ax.set_xlim(0, nsteps)
    ax.set_ylabel("lnprob")

    axes[-1].set_xlabel("step number")
    fig.suptitle(event_name)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    fig.savefig(path + "posteriors/" + event_name + f"_{burnin_or_post}_chain.png") # Updated save path
    plt.close(fig)


def corner_post(self, samples, event_name, path, truths, fisher_covariance=None, fisher_uncertainties=None, log_param_names=None):
    """Create a corner plot of the posterior samples, with Fisher uncertainty lines and ellipses.

    Parameters
    ----------
    samples : array_like
        Posterior samples with shape ``(nsamples, ndim)``.
    event_name : str
        Name used for the output file.
    path : str
        Directory where the figure will be saved.
    truths : dict
        Dictionary containing the true parameter values.
    fisher_covariance : array_like or None, optional
        Fisher covariance matrix (model_covariance, in linear space).
    fisher_uncertainties : array_like or None, optional
        1-sigma Fisher uncertainties for each parameter (in linear space).
    log_param_names : list or None, optional
        List of parameter names that are log-transformed.

    Notes
    -----
    The 68% confidence ellipse for each 2D parameter pair is drawn using the Fisher covariance matrix. The ellipse is defined by:
        (x - x0, y - y0)^T @ Sigma^{-1} @ (x - x0, y - y0) = k^2
    where Sigma is the 2x2 Fisher covariance submatrix, (x0, y0) is the truth, and k^2 = 2.30 for a 68% confidence region in 2D. The axes of the ellipse are given by the eigenvalues and eigenvectors of Sigma, and the ellipse is centered at the truth value.

    Returns
    -------
    None
    """
    import matplotlib.patches as mpatches
    if self.LOM_enabled:
        labels = [
            "s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN", "i", "phase", "period"
        ]
        true_params = truths["params"]
    else:
        labels = ["s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN"]
        true_params = truths["params"][:9]
    ndim = len(labels)
    if log_param_names is None:
        log_param_names = ["s", "q", "rho", "period"]

    fig = corner.corner(samples, labels=labels, truths=true_params)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # 1D: Add solid blue vertical lines at truth, and at truth ± Fisher uncertainty
    for i in range(ndim):
        ax = axes[i, i]
        truth = true_params[i]
        sigma = None if fisher_uncertainties is None else fisher_uncertainties[i]
        param_name = labels[i]
        truth_text = ""

        if truth is not None:
            ax.axvline(truth, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
            truth_text += f"True: {true_params[i]:.2f}"

        if sigma is not None:
            # Plot in physical space (corner plots are already in physical space)
            ax.axvline(truth + sigma, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(truth - sigma, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
            # The blue text for the truth and Fisher uncertainty
            truth_text += f" $\pm$ {fisher_uncertainties[i]:.2f}"

        # values labels
        p_16, p_50, p_84 = np.percentile(samples[:, i], [16, 50, 84])
        upper_unc = p_84 - p_50
        lower_unc = p_50 - p_16

        # The black text for your posterior results
        post_text = f"${p_50:.2f}^{{+{upper_unc:.2f}}}_{{-{lower_unc:.2f}}}$"

        # Place the black text (samples) near the top center
        ax.text(0.5, 1.05, post_text, color="black", ha='center', va='center', transform=ax.transAxes)

        # Place the blue text (Truth) just below it
        ax.text(0.5, 1.2, truth_text, color="blue", ha='center', va='center', transform=ax.transAxes)

    # 2D: Add solid blue cross-bars and 68% confidence ellipse
    if fisher_covariance is not None:
        k2 = 2.30  # 68% confidence region in 2D
        for i in range(ndim):
            for j in range(i):
                ax = axes[i, j]
                x0, y0 = true_params[j], true_params[i]
                cov = np.array([
                    [fisher_covariance[j, j], fisher_covariance[j, i]],
                    [fisher_covariance[i, j], fisher_covariance[i, i]]
                ])
                # Draw cross-bars
                sigma_x = np.sqrt(fisher_covariance[j, j])
                sigma_y = np.sqrt(fisher_covariance[i, i])
                ax.axvline(x0, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
                ax.axhline(y0, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
                ax.axvline(x0 + sigma_x, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axvline(x0 - sigma_x, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axhline(y0 + sigma_y, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axhline(y0 - sigma_y, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                # Draw 68% confidence ellipse
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]
                width, height = 2 * np.sqrt(k2 * vals)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                ellipse = mpatches.Ellipse((x0, y0), width, height, angle=angle, edgecolor="blue", facecolor="none", linestyle=":", linewidth=1.5, alpha=0.5, zorder=10)
                ax.add_patch(ellipse)

    fig.suptitle(event_name, y=1.0)
    fig.savefig(path + "posteriors/" + event_name + "_corner.png")
    plt.close(fig)

    if "corner" in self.debug:
        print("debug Fit.corner_post: labels: ", labels)
        print("debug Fit.corner_post: truths: ", truths["params"])
        print("debug Fit.corner_post: event_name: ", event_name)
        print("debug Fit.corner_post: path: ", path)
