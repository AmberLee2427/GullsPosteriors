import sys
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
    truths,
    prange_linear,
    prange_log,
    normal,
    threads=1,
    event_name="",
    path="./",
    labels=None,
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
    # The new arguments to be passed to the log probability function
    log_prob_args = [event_obj, truths, prange_linear, prange_log, normal]

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

                self.plot_chain(sampler, event_name, path, labels=labels)

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

            self.plot_chain(sampler, event_name, path, labels=labels)

            steps += stepi
            count += 1

    return sampler


def run_burnin(
    self,
    nl,
    ndim,
    stepi,
    max_steps,
    log_prob_function,
    state,
    event_obj,
    truths,
    prange_linear,
    prange_log,
    p_unc,
    normal,
    threads=1,
    event_name="",
    path="./",
    labels=None,
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
    max_steps : int
        Maximum burn-in steps to run.
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
    threads : int, optional
        Number of worker processes.
    event_name : str, optional
        Prefix for saved diagnostic files.
    path : str, optional
        Directory where output is written.
    labels : list of str or None, optional
        Parameter labels.

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
    expansion_threshold = 0.3

    while steps < max_steps and no_expand < 2:
        state, _, _ = sampler.run_mcmc(state, stepi, progress=self.show_progress)

        np.save(path + f"posteriors/{event_name}_burnin_samples.npy", sampler.flatchain)
        np.save(path + f"posteriors/{event_name}_burnin_lnprob.npy", sampler.flatlnprobability)
        np.save(path + f"posteriors/{event_name}_burnin_state.npy", state)

        self.plot_chain(sampler, f"{event_name}_burnin", path, labels=labels)

        # Update for emcee 3.x API
        positions = state
        expanded = False

        for j, idx in enumerate(log_indices):
            frac = np.mean((positions[:, idx] < 0.05) | (positions[:, idx] > 0.95))
            if frac > expansion_threshold:
                old_range = prange_log[j]
                prange_log[j] *= 1.2
                p_unc[idx] *= 1.2
                new_range = prange_log[j]
                center = truths[idx]
                old_min = np.log10(center) - old_range / 2.0
                new_min = np.log10(center) - new_range / 2.0
                phys = old_min + positions[:, idx] * old_range
                positions[:, idx] = (phys - new_min) / new_range
                expanded = True

        for j, idx in enumerate(lin_indices):
            frac = np.mean((positions[:, idx] < 0.05) | (positions[:, idx] > 0.95))
            if frac > expansion_threshold:
                old_range = prange_linear[j]
                prange_linear[j] *= 1.2
                p_unc[idx] *= 1.2
                new_range = prange_linear[j]
                center = truths[idx]
                old_min = center - old_range / 2.0
                new_min = center - new_range / 2.0
                phys = old_min + positions[:, idx] * old_range
                positions[:, idx] = (phys - new_min) / new_range
                expanded = True

        if expanded:
            log_prob, _ = sampler.compute_log_prob(positions)
            state = positions  # Update for emcee 3.x API
            no_expand = 0
        else:
            no_expand += 1

        steps += stepi

    if pool_ctx is not None:
        pool_ctx.close()

    return state, p_unc, prange_linear, prange_log


def lnprob_transform(
    self, u, event, true, prange_linear, prange_log, normal=False
):
    """Convert unit-cube samples to log-probability values.

    Parameters
    ----------
    u : array_like
        Sample from the unit hypercube.
    event : Event
        Microlensing event used to compute the likelihood.
    true : array_like
        Reference parameter values defining the prior centres.
    prange_linear : array_like
        Linear prior widths.
    prange_log : array_like
        Logarithmic prior widths in dex.
    normal : bool, optional
        If ``True``, sample the priors using normal distributions.

    Returns
    -------
    float
        Log-probability of the transformed sample.
    """
    # Check if any walker is somehow outside the unit cube.
    for uu in u:
        if not (0.0 <= uu <= 1.0):
            return -np.inf

    # Call the new, improved prior_transform with the correct arguments
    theta = self.prior_transform(u, true, prange_linear, prange_log, normal)

    # Calculate the log probability (likelihood) with the transformed
    # parameters
    lp = self.lnprob(theta, event)

    return lp


def plot_chain(self, res, event_name, path, labels=None, pt_args=[]):
    """Plot the trace of the walkers.

    Parameters
    ----------
    res : emcee.EnsembleSampler
        Sampler containing the chain and log-probability.
    event_name : str
        Name used when saving the figure.
    path : str
        Directory where the image will be written.
    labels : list of str or None, optional
        Parameter labels for each dimension.
    pt_args : list, optional
        Present for backward compatibility and ignored.

    Returns
    -------
    None
    """

    chain = res.chain  # shape = (nwalkers, nsteps, ndim)
    # print('Debug Fit.plot_chain: chain shape: ', chain.shape)
    lnprobability = res.lnprobability  # shape = (nwalkers, nsteps)
    # print('Debug Fit.plot_chain: lnprobability shape: ', lnprobability.shape)
    ndim = chain.shape[2]
    nsteps = chain.shape[1]
    nwalkers = chain.shape[0]

    if labels is None:
        labels = [f"theta[{i}]" for i in range(ndim)]

    fig, axes = plt.subplots(ndim + 1, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        for j in range(nwalkers):
            ax.plot(chain[j, :, i], "k", alpha=0.1)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(labels[i])
        # ax.yaxis.set_label_coords(-0.1, 0.5)

    ax = axes[-1]
    for j in range(nwalkers):
        ax.plot(lnprobability[j], "r", alpha=0.3)
    ax.plot(lnprobability, "r", alpha=0.3)
    ax.set_xlim(0, nsteps)
    ax.set_ylabel("lnprob")

    axes[-1].set_xlabel("step number")
    fig.suptitle(event_name)
    fig.savefig(path + "posteriors/" + event_name + "_chain.png")
    plt.close(fig)


def corner_post(self, samples, event_name, path, truths):
    """Create a corner plot of the posterior samples.

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

    Returns
    -------
    None
    """

    if self.LOM_enabled:
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
        true_params = truths["params"]
    else:
        labels = ["s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN"]
        true_params = truths["params"][:9]  # Use only the first 9 truth values

    fig = corner.corner(samples, labels=labels, truths=true_params)
    plt.title(event_name)

    fig.savefig(path + "posteriors/" + event_name + "_corner.png")
    plt.close(fig)

    if "corner" in self.debug:
        print("debug Fit.corner_post: labels: ", labels)
        print("debug Fit.corner_post: truths: ", truths["params"])
        print("debug Fit.corner_post: event_name: ", event_name)
        print("debug Fit.corner_post: path: ", path)
