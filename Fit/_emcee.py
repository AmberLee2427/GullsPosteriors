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
    fisher_uncertainties_for_plotting=None, # Added this argument
    plot_chains=False,
    show_progress=False
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
    plot_chains : bool, optional
        If ``True``, plot the chains after each stepi.
    show_progress : bool, optional
        If ``True``, show progress bars during sampling.

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

    # Set the plot chain and show progress attributes
    self.plot_chains = plot_chains
    self.show_progress = show_progress

    # Set up arguments based on which log probability function we're using
    # lnprob_transform expects: (self, u, event, truths_dict, prange_linear, prange_log, normal, fisher_uncertainties_for_prior)
    # lnprob expects: (self, theta, event)
    if log_prob_function == self.lnprob_transform:
        # Using unit-cube priors with transform
        log_prob_args = [event_obj, truths, prange_linear, prange_log, normal, fisher_uncertainties_for_prior]
    else:
        # Using direct physical space priors
        log_prob_args = [event_obj]

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
                # In emcee 3.x, run_mcmc returns a State object
                state = sampler.run_mcmc(state, stepi, progress=self.show_progress)
                flatchain = sampler.flatchain
                flatlnprobability = sampler.flatlnprobability
                
                
                # Extract blobs (flux parameters) if available
                # Use get_blobs() instead of deprecated .blobs property
                try:
                    blobs_data = sampler.get_blobs(flat=False)
                except AttributeError:
                    blobs_data = None
                
                if blobs_data is not None and len(blobs_data) > 0:
                    print(f"DEBUG: blobs_data type: {type(blobs_data)}")
                    print(f"DEBUG: blobs_data shape: {blobs_data.shape}")
                    print(f"DEBUG: blobs_data dtype: {blobs_data.dtype}")
                    if blobs_data.size > 0:
                        print(f"DEBUG: First blob element type: {type(blobs_data.flat[0])}")
                        print(f"DEBUG: First blob element: {blobs_data.flat[0]}")
                    
                    # blobs_data has shape (nsteps, nwalkers) with dtype object
                    # Each element is a dict or None
                    # First, collect all unique keys from all blobs to determine columns
                    all_keys = set()
                    for step_blobs in blobs_data:
                        for walker_blob in step_blobs:
                            if walker_blob is not None and isinstance(walker_blob, dict):
                                all_keys.update(walker_blob.keys())
                    
                    print(f"DEBUG: Found {len(all_keys)} unique keys in blobs")
                    print(f"DEBUG: Keys: {sorted(all_keys)[:10] if len(all_keys) > 10 else sorted(all_keys)}")
                    
                    if len(all_keys) > 0:
                        # Sort keys for consistent ordering (Fs_0, FB_0, Fbaseline_0, Fs_1, ...)
                        sorted_keys = sorted(all_keys)
                        
                        # Extract values for each sample - must ensure consistent shape
                        blobs_list = []
                        for step_idx, step_blobs in enumerate(blobs_data):
                            for walker_idx, walker_blob in enumerate(step_blobs):
                                if walker_blob is not None and isinstance(walker_blob, dict):
                                    # Extract values in sorted key order, use NaN for missing keys
                                    row = []
                                    for key in sorted_keys:
                                        val = walker_blob.get(key, np.nan)
                                        # Debug first few values
                                        if step_idx == 0 and walker_idx == 0 and len(row) < 3:
                                            print(f"DEBUG: key={key}, val={val}, type={type(val)}")
                                        row.append(float(val))
                                else:
                                    # Rejected sample - all NaN
                                    row = [np.nan] * len(sorted_keys)
                                blobs_list.append(row)
                        
                        print(f"DEBUG: Created {len(blobs_list)} blob rows")
                        if len(blobs_list) > 0:
                            # Ensure all rows have same length before converting to array
                            row_lengths = [len(row) for row in blobs_list]
                            unique_lengths = set(row_lengths)
                            print(f"DEBUG: Unique row lengths: {unique_lengths}")
                            if len(unique_lengths) > 1:
                                print(f"WARNING: Inconsistent blob row lengths: {unique_lengths}")
                                print(f"Expected {len(sorted_keys)} columns")
                                # Pad or truncate rows to match expected length
                                for i, row in enumerate(blobs_list):
                                    if len(row) < len(sorted_keys):
                                        blobs_list[i] = row + [np.nan] * (len(sorted_keys) - len(row))
                                    elif len(row) > len(sorted_keys):
                                        blobs_list[i] = row[:len(sorted_keys)]
                            
                            print(f"DEBUG: Converting to numpy array with shape ({len(blobs_list)}, {len(blobs_list[0])})")
                            blobs_array = np.array(blobs_list, dtype=float)
                            print(f"DEBUG: Successfully created array with shape {blobs_array.shape}")
                            # Save both the array and the column names
                            np.save(
                                path + "posteriors/" + event_name + "_emcee_blobs.npy",
                                blobs_array,
                            )
                            # Save column names as separate file for easier loading
                            np.save(
                                path + "posteriors/" + event_name + "_emcee_blobs_keys.npy",
                                np.array(sorted_keys),
                            )

                # Save the samples
                np.save(
                    path + "posteriors/" + event_name + "_emcee_samples.npy",
                    flatchain,
                )
                np.save(
                    path + "posteriors/" + event_name + "_emcee_lnprob.npy",
                    flatlnprobability,
                )
                # Save state coordinates only (state object itself can't be saved when it has dict blobs)
                np.save(
                    path + "posteriors/" + event_name + "_emcee_state_coords.npy",
                    state.coords,
                )

                # Pass the actual truths['params'] and fisher_uncertainties to plot_chain
                # Also pass prange_linear, prange_log, and normal for detransform_theta
                if self.plot_chains:
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
            # In emcee 3.x, run_mcmc returns a State object
            state = sampler.run_mcmc(state, stepi, progress=self.show_progress)
            flatchain = sampler.flatchain
            flatlnprobability = sampler.flatlnprobability
            
            # Extract blobs (flux parameters) if available
            if hasattr(sampler, 'blobs') and sampler.blobs is not None and len(sampler.blobs) > 0:
                # sampler.blobs is list of [step][walker] dicts
                # Each dict has keys like: Fs_0, FB_0, Fbaseline_0, Fs_1, FB_1, etc.
                # First, collect all unique keys from all blobs to determine columns
                all_keys = set()
                for step_blobs in sampler.blobs:
                    for walker_blob in step_blobs:
                        if walker_blob is not None and isinstance(walker_blob, dict):
                            all_keys.update(walker_blob.keys())
                
                if len(all_keys) > 0:
                    # Sort keys for consistent ordering (Fs_0, FB_0, Fbaseline_0, Fs_1, ...)
                    sorted_keys = sorted(all_keys)
                    
                    # Extract values for each sample - must ensure consistent shape
                    blobs_list = []
                    for step_blobs in sampler.blobs:
                        for walker_blob in step_blobs:
                            if walker_blob is not None and isinstance(walker_blob, dict):
                                # Extract values in sorted key order, use NaN for missing keys
                                row = [float(walker_blob.get(key, np.nan)) for key in sorted_keys]
                            else:
                                # Rejected sample - all NaN
                                row = [np.nan] * len(sorted_keys)
                            blobs_list.append(row)
                    
                    if len(blobs_list) > 0:
                        # Ensure all rows have same length before converting to array
                        row_lengths = [len(row) for row in blobs_list]
                        if len(set(row_lengths)) > 1:
                            print(f"WARNING: Inconsistent blob row lengths: {set(row_lengths)}")
                            print(f"Expected {len(sorted_keys)} columns")
                            # Pad or truncate rows to match expected length
                            for i, row in enumerate(blobs_list):
                                if len(row) < len(sorted_keys):
                                    blobs_list[i] = row + [np.nan] * (len(sorted_keys) - len(row))
                                elif len(row) > len(sorted_keys):
                                    blobs_list[i] = row[:len(sorted_keys)]
                        
                        blobs_array = np.array(blobs_list, dtype=float)
                        # Save both the array and the column names
                        np.save(
                            path + "posteriors/" + event_name + "_emcee_blobs.npy",
                            blobs_array,
                        )
                        # Save column names as separate file for easier loading
                        np.save(
                            path + "posteriors/" + event_name + "_emcee_blobs_keys.npy",
                            np.array(sorted_keys),
                        )

            # Save the samples
            np.save(
                path + "posteriors/" + event_name + "_emcee_samples.npy",
                flatchain,
            )
            np.save(
                path + "posteriors/" + event_name + "_emcee_lnprob.npy",
                flatlnprobability,
            )
            # Save state coordinates only (state object itself can't be saved when it has dict blobs)
            np.save(
                path + "posteriors/" + event_name + "_emcee_state_coords.npy",
                state.coords,
            )

            if self.plot_chains:
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
    fisher_uncertainties_for_plotting=None,  # Added this argument
    plot_chains=False,
    show_progress=False
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
    plot_chains : bool, optional
        If ``True``, plot the chains after each stepi.
    show_progress : bool, optional
        If ``True``, show progress bars during sampling.

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

    # Set the plot chain and show progress attributes
    self.plot_chains = plot_chains
    self.show_progress = show_progress

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

    log_param_names = ["s", "q", "rho", "tE"]
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
        if self.plot_chains:
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
    # parameters (lnprob returns (log_prob, blobs))
    lp, blobs = self.lnprob(theta, event)

    return lp, blobs


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
    chain = res.chain  # shape = (nwalkers, nsteps, ndim) - Could be unit cube OR physical space
    lnprobability = res.lnprobability  # shape = (nwalkers, nsteps)
    ndim = chain.shape[2]
    nsteps = chain.shape[1]
    nwalkers = chain.shape[0]

    # Detect whether chain is in unit-cube space or physical space
    # Unit-cube chains have all values between 0 and 1
    # Physical chains have realistic parameter values
    chain_flat = chain.reshape(-1, ndim)
    is_unit_cube = np.all((chain_flat >= 0) & (chain_flat <= 1))
    
    print(f"Chain space detection: {'unit-cube' if is_unit_cube else 'physical'}")
    print(f"Chain value ranges: min={np.min(chain_flat, axis=0)}, max={np.max(chain_flat, axis=0)}")

    # Map truths/fisher arrays to the current parameter ordering (self.labels)
    current_labels = self.labels if hasattr(self, 'labels') and self.labels is not None else [f"theta[{i}]" for i in range(ndim)]
    full_labels_list = [
        "s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN", "i", "phase", "period"
    ]

    truths_full = None
    truths_arr = None
    if truths is not None:
        t = np.asarray(truths).reshape(-1)
        if t.shape[0] == 12:
            truths_full = t
            # Map to subset in current order
            idx_map = [full_labels_list.index(lbl) for lbl in current_labels]
            truths_arr = truths_full[idx_map]
        elif t.shape[0] == ndim:
            truths_arr = t
        else:
            raise ValueError(
                f"plot_chain: 'truths' length ({t.shape[0]}) must be either 12 (full set) or match chain ndim ({ndim})."
            )

    fisher_arr = None
    if fisher_uncertainties_for_plotting is not None:
        # Support dict mapping label->sigma
        if isinstance(fisher_uncertainties_for_plotting, dict):
            label_sigma = fisher_uncertainties_for_plotting
            fisher_arr = np.full(ndim, np.nan, dtype=float)
            for i, lbl in enumerate(current_labels):
                if lbl in label_sigma and label_sigma[lbl] is not None:
                    fisher_arr[i] = float(label_sigma[lbl])
        else:
            f = np.asarray(fisher_uncertainties_for_plotting).reshape(-1)
            # Build an aligned array of size ndim, NaN where not available
            fisher_arr = np.full(ndim, np.nan, dtype=float)
            base_labels = None
            base = None
            if f.shape[0] >= 12:
                # Assume first 12 are model params in canonical order; extra entries are flux
                base_labels = full_labels_list[:12]
                base = f[:12]
            elif f.shape[0] == 9:
                # Model-only uncertainties (no LOM)
                base_labels = full_labels_list[:9]
                base = f
            elif f.shape[0] == ndim:
                # Already aligned to current params
                fisher_arr = f
            else:
                raise ValueError(
                    f"plot_chain: 'fisher_uncertainties_for_plotting' length ({f.shape[0]}) is unsupported. "
                    "Provide a dict label->sigma, a vector of length ndim, 9 (model-only), or >=12 (model + possibly flux)."
                )

            if base_labels is not None:
                label_sigma = {lbl: float(val) for lbl, val in zip(base_labels, base)}
                for i, lbl in enumerate(current_labels):
                    if lbl in label_sigma:
                        fisher_arr[i] = label_sigma[lbl]

    if labels is None:
        labels = [f"theta[{i}]" for i in range(ndim)]
    else:
        if len(labels) != ndim:
            raise ValueError(
                f"plot_chain: labels length ({len(labels)}) does not match chain ndim ({ndim})."
            )

    # Handle unit-cube vs physical space plotting
    if is_unit_cube:
        # Chain is in unit-cube space - use the existing logic
        if truths_arr is not None:
            # Detransform the truth values to unit cube space
            if truths_full is None:
                raise ValueError("plot_chain: unit-cube chains require the full 12-parameter truths array for detransform.")
            u_truths_mapped = self.detransform_theta(
                truths_arr,
                truths_full,
                prange_linear,
                prange_log,
                normal,
                fisher_uncertainties_for_prior # USE THE CORRECT PRIOR DEFINITION
            )
            # do a sanity check that u_truths_mapped is an ndim-D array of ~0.5 values
            if u_truths_mapped.shape[0] != ndim:
                print(f"u_truths_mapped.shape: {u_truths_mapped.shape}")
                print(f"ndim: {ndim}")
                print(f'u_truths_mapped type: {type(u_truths_mapped)}')
                raise ValueError("u_truths_mapped is not an ndim-D array")
            if not np.allclose(u_truths_mapped, 0.5, atol=1e-6):
                print("WARNING: u_truths_mapped is not close to an array of 0.5 values")
                print(f"u_truths_mapped: {u_truths_mapped}")
                print(f"Max deviation from 0.5: {np.max(np.abs(u_truths_mapped - 0.5))}")

        if fisher_arr is not None and truths_arr is not None:
            # Create temporary arrays for truth +/- sigma for detransformation    
            theta_plus_sigma = np.array(truths_arr) + np.array(fisher_arr)
            theta_minus_sigma = np.array(truths_arr) - np.array(fisher_arr)

            # Detransform these values to unit cube space
            u_plus_sigma_mapped = self.detransform_theta(
                theta_plus_sigma,
                truths_full,
                prange_linear,
                prange_log,
                normal,
                fisher_uncertainties_for_prior  # None if not using
            )
            
            u_minus_sigma_mapped = self.detransform_theta(
                theta_minus_sigma,
                truths_full,
                prange_linear,
                prange_log,
                normal,
                fisher_uncertainties_for_prior
            )
        else:
            u_plus_sigma_mapped = None
            u_minus_sigma_mapped = None

        # Plot unit-cube chains
        fig, axes = plt.subplots(ndim + 1, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            for j in range(nwalkers):
                ax.plot(chain[j, :, i], "k", alpha=0.1)
            ax.set_xlim(0, nsteps)
            ax.set_ylabel(labels[i])
            ax.set_ylim(-0.1, 1.1) # Unit cube limits

            # Add horizontal line for the truth (in unit cube space)
            if truths_arr is not None:
                ax.axhline(u_truths_mapped[i], color="blue", linestyle="-", linewidth=1.5, alpha=0.7, label="Truth (Unit Cube)")

            # Add horizontal dashed lines for 1-sigma uncertainties (in unit cube space)
            if u_plus_sigma_mapped is not None:
                ax.axhline(u_plus_sigma_mapped[i], color="blue", linestyle="--", linewidth=1.0, alpha=0.7, label="Truth $\pm 1\sigma_{Fisher}$ (Unit Cube)")
                ax.axhline(u_minus_sigma_mapped[i], color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                
            # Add legend to the first subplot only to avoid clutter
            if i == 0 and truths_arr is not None:
                ax.legend(loc='best')

    else:
        # Chain is in physical space - plot directly
        fig, axes = plt.subplots(ndim + 1, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            for j in range(nwalkers):
                ax.plot(chain[j, :, i], "k", alpha=0.1)
            ax.set_xlim(0, nsteps)
            ax.set_ylabel(labels[i])
            
            # Set reasonable y-limits based on the data range
            param_values = chain[:, :, i].flatten()
            y_min, y_max = np.percentile(param_values, [1, 99])
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            # Add horizontal line for the truth (in physical space)
            if truths_arr is not None:
                ax.axhline(truths_arr[i], color="blue", linestyle="-", linewidth=1.5, alpha=0.7, label="Truth")

            # Add horizontal dashed lines for 1-sigma uncertainties (in physical space)
            if fisher_arr is not None and truths_arr is not None:
                ax.axhline(truths_arr[i] + fisher_arr[i], color="blue", linestyle="--", linewidth=1.0, alpha=0.7, label="Truth $\pm 1\sigma_{Fisher}$")
                ax.axhline(truths_arr[i] - fisher_arr[i], color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                
            # Add legend to the first subplot only to avoid clutter
            if i == 0 and truths_arr is not None:
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


def corner_post(
    self,
    samples,
    event_name,
    path, truths,
    fisher_covariance=None,
    fisher_covariance_schur=None,
    fisher_uncertainties=None,
    log_param_names=None,
    k2=1,
    return_figure=False,
    use_schur=False):
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
        Full (block from inverse) Fisher covariance matrix (model parameters only) in plotting space.
    fisher_covariance_schur : array_like or None, optional
        Schur-complement covariance alternative for the model parameter block.
    fisher_uncertainties : array_like or None, optional
        1-sigma uncertainties (will be derived from the selected covariance if not provided).
    log_param_names : list or None, optional
        List of parameter names that are log-transformed.
    k2 : float, optional
        The k² value for the confidence ellipse. Default is 1 (39% confidence).
    return_figure : bool, optional
        If True, return the matplotlib figure object instead of saving to file.
    use_schur : bool, optional
        When True and ``fisher_covariance_schur`` is provided, the Schur covariance
        (and its diagonal uncertainties) are used for all uncertainty lines and ellipses.

    Returns
    -------
    matplotlib.figure.Figure or None
        If return_figure=True, returns the matplotlib figure object.
        If return_figure=False, returns None (saves to file instead).

    Notes
    -----
    The 68% confidence ellipse for each 2D parameter pair is drawn using the 
    Fisher covariance matrix. The ellipse is defined by:
        (x - x0, y - y0)^T @ Sigma^{-1} @ (x - x0, y - y0) = k^2
    where Sigma is the 2x2 Fisher covariance submatrix, (x0, y0) is the truth.
    For a bivariate normal the quadratic form on the left follows a χ² 
    distribution with ν = 2 degrees of freedom
    Therefore,
        P [ inside ellipse ] = Fχ²₂,
    where Fχ²₂ is the cumulative distribution function (CDF) of the χ² 
    distribution with ν = 2 degrees of freedom.
    * k² = 1 → P = 1 - e^(–½) ≈ 0.393  (≈ 39 %)
    * k² = 2.30 → P ≈ 0.683  (the usual “1 σ” ≃ 68 %)
    * k² = 4.61 → P ≈ 0.954  (the usual “2 σ” ≃ 95 %)
    * k² = 9.21 → P ≈ 0.997  (the usual “3 σ” ≃ 99.7 %)
    So:
    * Setting k² = 1 draws the ellipse that encloses the region where the 
    Mahalanobis distance is ≤ 1; that region contains about 39 % of the 
    probability mass in two dimensions.
    * Setting k² = 2.30 draws the ellipse that encloses 68.3% of the mass--the 
    2-D analogue of the familiar 1 σ (68 %) interval in 1-D.
    
    The axes of the ellipse 
    are given by the eigenvalues and eigenvectors of Sigma, and the ellipse is 
    centered at the truth value.

    Returns
    -------
    None
    """
    import matplotlib.patches as mpatches

    # Select active covariance (Schur if requested and available)
    active_cov = None
    if use_schur and (fisher_covariance_schur is not None):
        active_cov = fisher_covariance_schur
    else:
        active_cov = fisher_covariance

    # If uncertainties not supplied, derive from whichever covariance is active
    if fisher_uncertainties is None and active_cov is not None:
        fisher_uncertainties = np.sqrt(np.diag(active_cov))
    
    # Determine the number of parameters from samples shape
    ndim = samples.shape[1]
    
    # Set up labels based on actual number of parameters
    if ndim == 12 or (hasattr(self, 'LOM_enabled') and self.LOM_enabled):
        labels = [
            r"$\log_{10}s$", r"$\log_{10}q$", r"$\log_{10}{\rho}$", 
            r"$u_0$", r"$\alpha$", r"$t_0$", r"$\log_{10}t_E$", 
            r"$\pi_{EE}$", r"$\pi_{EN}$", 
            r"$i$", r"$\phi$", r"$\log_{10}{period}$"
        ]
        true_params = truths["params"]
    else:
        labels = [
            r"$\log_{10}s$", r"$\log_{10}q$", r"$\log_{10}{\rho}$", 
            r"$u_0$", r"$\alpha$", r"$t_0$", r"$\log_{10}t_E$", 
            r"$\pi_{EE}$", r"$\pi_{EN}$"
        ]
        true_params = truths["params"][:9]
    
    # Trim labels and truths to match actual number of parameters
    labels = labels[:ndim]
    true_params = true_params[:ndim]
    
    # Convert fisher_uncertainties to array format if it's a dict
    if fisher_uncertainties is not None and isinstance(fisher_uncertainties, dict):
        full_labels_list = ["s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN", "i", "phase", "period"]
        fisher_arr = np.full(ndim, np.nan, dtype=float)
        label_map = {
            r"$\log_{10}s$": "s", r"$\log_{10}q$": "q", r"$\log_{10}{\rho}$": "rho",
            r"$u_0$": "u0", r"$\alpha$": "alpha", r"$t_0$": "t0", r"$\log_{10}t_E$": "tE",
            r"$\pi_{EE}$": "piEE", r"$\pi_{EN}$": "piEN",
            r"$i$": "i", r"$\phi$": "phase", r"$\log_{10}{period}$": "period"
        }
        for i in range(ndim):
            simple_label = label_map.get(labels[i])
            if simple_label and simple_label in fisher_uncertainties:
                fisher_arr[i] = float(fisher_uncertainties[simple_label])
        fisher_uncertainties = fisher_arr
    elif fisher_uncertainties is not None:
        # Already an array, just ensure it's numpy array
        fisher_uncertainties = np.asarray(fisher_uncertainties).reshape(-1)
    
    if log_param_names is None:
        log_param_names = ["s", "q", "rho", "tE", "period"]
    # replace tE with t_E
    log_param_names = [name.replace("tE", "t_E") for name in log_param_names]

    # Samples are already in the correct space (log for log params, linear for linear params)
    # No transformation needed - just copy the samples
    processed_samples = samples.copy()

    # Drop the first half of the samples
    nsteps = processed_samples.shape[0]
    processed_samples = processed_samples[nsteps//2:]
    
    # Convert truth values to the same space as samples (log space for log parameters)
    plot_truths = true_params.copy()
    log_indicies = []
    # replace the logs
    for log_parameter in log_param_names:
        print(f"Checking for log parameter: {log_parameter}")
        for i in range(ndim):
            if log_parameter in labels[i] and "log" in labels[i]:
                print(f"Replacing {labels[i]} with log10")
                plot_truths[i] = np.log10(true_params[i]) if true_params[i] > 0 else np.nan
                log_indicies.append(i)

    fig = corner.corner(processed_samples, labels=labels, truths=plot_truths)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # 1D: Add solid blue vertical lines at truth, and at truth ± Fisher uncertainty
    for i, truth in enumerate(plot_truths):
        ax = axes[i, i]
        truth = plot_truths[i]  # Already in correct space
        sigma = None if fisher_uncertainties is None else fisher_uncertainties[i]
        param_name = labels[i]
        truth_text = f"{param_name} = "

        if truth is not None and not np.isnan(truth):
            # Plot truth line (already in correct space)
            ax.axvline(truth, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
            
            # For log parameters: display truth in log space to match the plot
            truth_text += f"{truth:.4f}"  # truth is already in log space

        if sigma is not None:
            # Fisher uncertainties are already in the correct space from Data class
            # Plot uncertainty lines directly
            ax.axvline(truth + sigma, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(truth - sigma, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
            
            # For all parameters: display uncertainty in the same space as the plot
            truth_text += f" $\pm$ {sigma:.4f}"

        # Calculate posterior statistics from the processed samples (which are in correct space)
        p_16, p_50, p_84 = np.percentile(processed_samples[:, i], [16, 50, 84])
        upper_unc = p_84 - p_50
        lower_unc = p_50 - p_16

        # The black text for your posterior results
        post_text = f"{param_name} = ${p_50:.4f}^{{+{upper_unc:.4f}}}_{{-{lower_unc:.4f}}}$"

        # Place the black text (samples) near the top center
        ax.text(0.5, 1.05, post_text, color="black", ha='center', va='center', transform=ax.transAxes)

        # Place the blue text (Truth) just below it
        ax.text(0.5, 1.15, truth_text, color="blue", ha='center', va='center', transform=ax.transAxes)

    # 2D: Add solid blue cross-bars and 68% confidence ellipse
    if active_cov is not None:
        for i in range(ndim):
            for j in range(i):
                ax = axes[i, j]
                # Use plot truths which are already in correct space
                x0, y0 = plot_truths[j], plot_truths[i]
                
                # Extract 2x2 covariance submatrix
                cov = np.array([
                    [active_cov[j, j], active_cov[j, i]],
                    [active_cov[i, j], active_cov[i, i]]
                ])

                # Draw cross-bars using Fisher uncertainties (already in correct space)
                sigma_x = np.sqrt(active_cov[j, j])
                sigma_y = np.sqrt(active_cov[i, i])

                ax.axvline(x0, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
                ax.axhline(y0, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
                ax.axvline(x0 + sigma_x, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axvline(x0 - sigma_x, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axhline(y0 + sigma_y, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axhline(y0 - sigma_y, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
                
                # Draw confidence ellipse
                vals, vecs = np.linalg.eigh(cov)  # Compute eigenvalues and eigenvectors
                order = vals.argsort()[::-1]  # Sort eigenvalues in descending order
                vals = vals[order]
                vecs = vecs[:, order]  # Sort eigenvectors in the sameorder
                width, height = 2 * np.sqrt(k2 * vals)  # Compute width and height of the ellipse
                                                        # a = √(k² λ₁), b = √(k² λ₂)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))  # Compute the angle of the ellipse
                ellipse = mpatches.Ellipse(
                    (x0, y0), 
                    width, 
                    height, 
                    angle=angle, 
                    edgecolor="blue", 
                    facecolor="none", 
                    linestyle="-", 
                    linewidth=1.5, 
                    alpha=0.5, 
                    zorder=10
                )
                ax.add_patch(ellipse)

    fig.suptitle(event_name, y=1.0)
    
    if return_figure:
        # Return the figure object for display in notebook
        return fig
    else:
        # Save to file and close (original behavior)
        # Create directory if it doesn't exist
        import os
        output_dir = path + "posteriors/"
        os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(output_dir + event_name + "_corner.png")
        plt.close(fig)

    if "corner" in self.debug:
        print("debug Fit.corner_post: labels: ", labels)
        print("debug Fit.corner_post: truths: ", truths["params"])
        print("debug Fit.corner_post: event_name: ", event_name)
        print("debug Fit.corner_post: path: ", path)
