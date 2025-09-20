# In Fit/_dynesty.py
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from numpy import sqrt
import multiprocessing as mp
import pickle

def prior_transform(
    self, u, truths_array, prange_linear, prange_log, normal=False, fisher_uncertainties=None
):
    """Map unit-cube samples to physical parameters.

    Parameters
    ----------
    u : array_like
        Samples from the unit hypercube with shape ``(nwalkers, ndim)`` or
        ``(ndim,)``.
    truths_array : array_like
        Reference parameter values for the complete 12-parameter model (as an array).
    prange_linear : array_like
        Linear prior widths for the current model parameters.
    prange_log : array_like
        Logarithmic prior widths for the current model parameters.
    normal : bool, optional
        If ``True``, draw from normal rather than uniform distributions (in unit cube space, mean=0.5, sigma=0.25).
    fisher_uncertainties : array_like or None, optional
        1-sigma Fisher uncertainties for each parameter (in linear space for linear params, log space for log params).
        If provided, the prior region is [truth-4σ, truth+4σ] (or log equivalent). The 2σ region maps to [0.25, 0.75] in the unit cube.
        For normal, samples are drawn from N(0.5, 0.25) in the unit cube, then mapped.
        If not provided, prange_log/prange_linear are used as before.

    Notes
    -----
    For both uniform and normal priors:
      - [0, 0.25, 0.5, 0.75, 1.0] in the unit cube maps to [truth-4σ, truth-2σ, truth, truth+2σ, truth+4σ] (linear), or log equivalents for log parameters.
      - For normal, samples are drawn from N(0.5, 0.25) in the unit cube, then mapped as above.
      - For log parameters, handle asymmetry if σ+ ≠ σ−.
    No manual truncation is needed; the unit cube bounds do this.

    Returns
    -------
    ndarray
        Array of transformed parameters with the same shape as ``u``.
    """
    theta = np.zeros_like(u)  # Output array, same shape as u

    # Use the labels stored in the Fit object to determine the current
    # parameter set
    current_labels = self.labels

    # Define which parameter names are log-transformed
    log_param_names_base = ["s", "q", "rho", "tE"]
    if self.LOM_enabled:
        log_param_names = log_param_names_base + ["period"]
    else:
        log_param_names = log_param_names_base

    u_log_indices = [
        i for i, label in enumerate(current_labels) if label in log_param_names
    ]
    u_linear_indices = [
        i for i, label in enumerate(current_labels) if label not in log_param_names
    ]

    # Get indices for log and linear params within the *full 12-param truth
    # array*
    full_labels_list = [
        "s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN", "i", "phase", "period"
    ]
    true_log_indices = [full_labels_list.index(name) for name in log_param_names]
    current_linear_labels = [label for label in current_labels if label not in log_param_names]
    true_linear_indices = [full_labels_list.index(name) for name in current_linear_labels]

    true_log_values = truths_array[true_log_indices] # Use truths_array directly
    true_linear_values = truths_array[true_linear_indices] # Use truths_array directly

    # --- Transform log parameters ---
    if u.ndim == 1:  # Single sample
        for i, u_idx in enumerate(u_log_indices):
            true_val = true_log_values[i]
            prange_val = prange_log[i]
            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                log_true = np.log(true_val)  # Use natural log
                # Ensure log_sigma is calculated correctly for positive/negative deviations
                # For simplicity, assuming symmetric log_sigma based on (true_val + sigma)
                # A more robust approach might consider log(true_val - sigma) if applicable
                log_sigma = np.log(true_val + sigma) - log_true  # Use natural log
                if normal:
                    loc = log_true
                    scale = 4 * log_sigma # Scale for 4-sigma range in log space
                    theta[u_idx] = np.exp(norm.ppf(u[u_idx], loc=loc, scale=scale))  # Use exp for natural log
                else:
                    min_log = log_true - 4 * log_sigma
                    max_log = log_true + 4 * log_sigma
                    theta[u_idx] = np.exp(min_log + (max_log - min_log) * u[u_idx])  # Use exp for natural log
            elif normal:
                loc = np.log(true_val)  # Use natural log
                scale = prange_val / 2.0 # Scale for prange_val/2 width in log space
                theta[u_idx] = np.exp(norm.ppf(u[u_idx], loc=loc, scale=scale))  # Use exp for natural log
            else:
                min_log = np.log(true_val) - prange_val / 2.0  # Use natural log
                max_log = np.log(true_val) + prange_val / 2.0  # Use natural log
                theta[u_idx] = np.exp(min_log + (max_log - min_log) * u[u_idx])  # Use exp for natural log
    else:  # Multiple samples (walkers)
        for i, u_idx in enumerate(u_log_indices):
            true_val = true_log_values[i]
            prange_val = prange_log[i]
            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                log_true = np.log(true_val)  # Use natural log
                log_sigma = np.log(true_val + sigma) - log_true  # Use natural log
                if normal:
                    loc = log_true
                    scale = 4 * log_sigma
                    theta[:, u_idx] = np.exp(norm.ppf(u[:, u_idx], loc=loc, scale=scale))  # Use exp for natural log
                else:
                    min_log = log_true - 4 * log_sigma
                    max_log = log_true + 4 * log_sigma
                    theta[:, u_idx] = np.exp(min_log + (max_log - min_log) * u[:, u_idx])  # Use exp for natural log
            elif normal:
                loc = np.log(true_val)  # Use natural log
                scale = prange_val / 2.0
                theta[:, u_idx] = np.exp(norm.ppf(
                    u[:, u_idx], loc=loc, scale=scale
                ))  # Use exp for natural log
            else:
                min_log = np.log(true_val) - prange_val / 2.0  # Use natural log
                max_log = np.log(true_val) + prange_val / 2.0  # Use natural log
                theta[:, u_idx] = np.exp(
                    min_log + (max_log - min_log) * u[:, u_idx]
                )  # Use exp for natural log

    # --- Transform linear parameters ---
    # prange_linear is already correctly sized
    if u.ndim == 1:
        for i, u_idx in enumerate(u_linear_indices):
            true_val = true_linear_values[i]
            prange_val = prange_linear[i]
            label = current_linear_labels[i]
            is_angle = label in ["alpha", "i", "phase"]
            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                if is_angle and normal:
                    # max_sigma is a safeguard to prevent too wide a distribution for angles
                    # It ensures the 4-sigma range doesn't exceed 2*pi / 2 (i.e., pi)
                    # This prevents extreme values that would wrap multiple times
                    max_sigma = (np.pi - 1e-6) / 8 # This means 4*sigma_max = (np.pi - 1e-6)/2
                    if sigma > max_sigma:
                        sigma = max_sigma
                if normal:
                    if is_angle:
                        # For angles, we sample a deviation from true_val
                        # The prior is effectively a normal distribution centered at true_val
                        # The scale is 4*sigma for Fisher, or prange_val/2 for non-Fisher
                        theta[u_idx] = norm.ppf(u[u_idx], loc=true_val, scale=4 * sigma)
                    else:
                        loc = true_val
                        scale = 4 * sigma
                        theta[u_idx] = norm.ppf(u[u_idx], loc=loc, scale=scale)
                else: # Uniform distribution with Fisher uncertainties
                    min_linear = true_val - 4 * sigma
                    max_linear = true_val + 4 * sigma
                    theta[u_idx] = min_linear + (max_linear - min_linear) * u[u_idx]
            elif normal:
                if is_angle:
                    theta[u_idx] = norm.ppf(u[u_idx], loc=true_val, scale=prange_val / 2.0)
                else:
                    loc = true_val
                    scale = prange_val / 2.0
                    theta[u_idx] = norm.ppf(u[u_idx], loc=loc, scale=scale)
            else: # Uniform distribution without Fisher uncertainties
                min_linear = true_val - prange_val / 2.0
                max_linear = true_val + prange_val / 2.0
                theta[u_idx] = (
                    min_linear + (max_linear - min_linear) * u[u_idx]
                )
    else: # Multiple samples (walkers)
        for i, u_idx in enumerate(u_linear_indices):
            true_val = true_linear_values[i]
            prange_val = prange_linear[i]
            label = current_linear_labels[i]
            is_angle = label in ["alpha", "i", "phase"]
            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                if is_angle and normal:
                    max_sigma = (np.pi - 1e-6) / 8
                    if sigma > max_sigma:
                        sigma = max_sigma
                if normal:
                    if is_angle:
                        theta[:, u_idx] = norm.ppf(u[:, u_idx], loc=true_val, scale=4 * sigma)
                    else:
                        loc = true_val
                        scale = 4 * sigma
                        theta[:, u_idx] = norm.ppf(u[:, u_idx], loc=loc, scale=scale)
                else: # Uniform distribution with Fisher uncertainties
                    min_linear = true_val - 4 * sigma
                    max_linear = true_val + 4 * sigma
                    theta[:, u_idx] = min_linear + (max_linear - min_linear) * u[:, u_idx]
            elif normal:
                if is_angle:
                    theta[:, u_idx] = norm.ppf(u[:, u_idx], loc=true_val, scale=prange_val / 2.0)
                else:
                    loc = true_val
                    scale = prange_val / 2.0
                    theta[:, u_idx] = norm.ppf(u[:, u_idx], loc=loc, scale=scale)
            else: # Uniform distribution without Fisher uncertainties
                min_linear = true_val - prange_val / 2.0
                max_linear = true_val + prange_val / 2.0
                theta[:, u_idx] = (
                    min_linear + (max_linear - min_linear) * u[:, u_idx]
                )

    return theta

def detransform_theta(self, theta, truths_array, prange_linear, prange_log, normal=False, fisher_uncertainties=None):
    """Map physical parameters to unit-cube samples.

    Parameters
    ----------
    theta : array_like
        Array of transformed parameters with the same shape as ``u``.
    truths_array : array_like
        Reference ("truth") parameter values for the complete 12-parameter model (as an array).
    prange_linear : array_like
        Linear prior widths for the current model parameters.
    prange_log : array_like
        Logarithmic prior widths for the current model parameters.
    normal : bool, optional
        If ``True``, draw from normal rather than uniform distributions (in unit cube space, mean=0.5, sigma=0.25).
    fisher_uncertainties : array_like or None, optional
        1-sigma Fisher uncertainties for each parameter (in linear space for linear params, log space for log params).
        If provided, the prior region is [truth-2σ, truth+2σ] (or log equivalent). The 1σ region maps to [0.25, 0.75] in the unit cube.
        For normal, samples are drawn from N(0.5, 0.25) in the unit cube, then mapped.
    
    Returns
    -------
    u : array_like
        Unit-cube samples with the same shape as ``theta``.
    """
    u = np.zeros_like(theta)

    # Use the labels stored in the Fit object to determine the current
    # parameter set
    current_labels = self.labels

    # Define which parameter names are log-transformed
    log_param_names_base = ["s", "q", "rho", "tE"]

    # Complete the log_param_names logic
    if self.LOM_enabled:
        log_param_names = log_param_names_base + ["period"]
    else:
        log_param_names = log_param_names_base

    u_log_indices = [
        i for i, label in enumerate(current_labels) if label in log_param_names
    ]
    u_linear_indices = [
        i for i, label in enumerate(current_labels) if label not in log_param_names
    ]

    # Get indices for log and linear params within the *full 12-param truth array*
    full_labels_list = [
        "s", "q", "rho", "u0", "alpha", "t0", "tE", "piEE", "piEN", "i", "phase", "period"
    ]
    true_log_indices = [full_labels_list.index(name) for name in log_param_names]
    current_linear_labels = [label for label in current_labels if label not in log_param_names]
    true_linear_indices = [full_labels_list.index(name) for name in current_linear_labels]

    true_log_values = truths_array[true_log_indices] # Use truths_array directly
    true_linear_values = truths_array[true_linear_indices] # Use truths_array directly

    # --- Invert log parameters ---
    if theta.ndim == 1:  # Single sample
        for i, u_idx in enumerate(u_log_indices):
            true_val = true_log_values[i]
            prange_val = prange_log[i]
            log_theta = np.log(theta[u_idx])  # Use natural log
            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                log_true = np.log(true_val)  # Use natural log
                log_sigma = np.log(true_val + sigma) - log_true  # Use natural log
                if normal:
                    loc = log_true
                    scale = 4 * log_sigma
                    u[u_idx] = norm.cdf(log_theta, loc=loc, scale=scale)
                else:
                    min_log = log_true - 4 * log_sigma
                    max_log = log_true + 4 * log_sigma
                    u[u_idx] = (log_theta - min_log) / (max_log - min_log)
            elif normal:
                loc = np.log(true_val)  # Use natural log
                scale = prange_val / 2.0
                u[u_idx] = norm.cdf(log_theta, loc=loc, scale=scale)
            else:
                min_log = np.log(true_val) - prange_val / 2.0  # Use natural log
                max_log = np.log(true_val) + prange_val / 2.0  # Use natural log
                u[u_idx] = (log_theta - min_log) / (max_log - min_log)
    else:  # Multiple samples (walkers)
        for i, u_idx in enumerate(u_log_indices):
            true_val = true_log_values[i]
            prange_val = prange_log[i]
            log_theta = np.log(theta[:, u_idx])  # Use natural log
            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                log_true = np.log(true_val)  # Use natural log
                log_sigma = np.log(true_val + sigma) - log_true  # Use natural log
                if normal:
                    loc = log_true
                    scale = 4 * log_sigma
                    u[:, u_idx] = norm.cdf(log_theta, loc=loc, scale=scale)
                else:
                    min_log = log_true - 4 * log_sigma
                    max_log = log_true + 4 * log_sigma
                    u[:, u_idx] = (log_theta - min_log) / (max_log - min_log)
            elif normal:
                loc = np.log(true_val)  # Use natural log
                scale = prange_val / 2.0
                u[:, u_idx] = norm.cdf(log_theta, loc=loc, scale=scale)
            else:
                min_log = np.log(true_val) - prange_val / 2.0  # Use natural log
                max_log = np.log(true_val) + prange_val / 2.0  # Use natural log
                u[:, u_idx] = (log_theta - min_log) / (max_log - min_log)

    # --- Invert linear parameters ---
    def angle_diff(a, b):
        """Calculates the shortest angular difference between two angles."""
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    if theta.ndim == 1:
        for i, u_idx in enumerate(u_linear_indices):
            true_val = true_linear_values[i]
            prange_val = prange_linear[i]
            theta_val = theta[u_idx]
            label = current_linear_labels[i]
            is_angle = label in ["alpha", "i", "phase"]

            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                if is_angle and normal:
                    max_sigma = (np.pi - 1e-6) / 8
                    if sigma > max_sigma:
                        sigma = max_sigma
                    u_val = norm.cdf(theta_val, loc=true_val, scale=4 * sigma)
                elif normal: # Non-angle, normal with Fisher
                    loc = true_val
                    scale = 4 * sigma
                    u_val = norm.cdf(theta_val, loc=loc, scale=scale)
                else: # Uniform distribution with Fisher uncertainties
                    min_linear = true_val - 4 * sigma
                    max_linear = true_val + 4 * sigma
                    u_val = (theta_val - min_linear) / (max_linear - min_linear)
            elif normal: # Normal distribution without Fisher uncertainties
                if is_angle:
                    delta = angle_diff(theta_val, true_val)
                    u_val = norm.cdf(delta, loc=0.0, scale=prange_val / 2.0)
                else:
                    loc = true_val
                    scale = prange_val / 2.0
                    u_val = norm.cdf(theta_val, loc=loc, scale=scale)
            else: # Uniform distribution without Fisher uncertainties
                if is_angle:
                    min_linear = true_val - prange_val / 2.0
                    max_linear = true_val + prange_val / 2.0
                    u_val = (angle_diff(theta_val, true_val) + prange_val / 2.0) / prange_val
                else:
                    min_linear = true_val - prange_val / 2.0
                    max_linear = true_val + prange_val / 2.0
                    u_val = (theta_val - min_linear) / (max_linear - min_linear)
            u[u_idx] = u_val
    else: # Multiple samples (walkers)
        for i, u_idx in enumerate(u_linear_indices):
            true_val = true_linear_values[i]
            prange_val = prange_linear[i]
            theta_val = theta[:, u_idx]
            label = current_linear_labels[i]
            is_angle = label in ["alpha", "i", "phase"]

            if fisher_uncertainties is not None:
                sigma = fisher_uncertainties[u_idx]
                if is_angle and normal:
                    max_sigma = (np.pi - 1e-6) / 8
                    if sigma > max_sigma:
                        sigma = max_sigma
                    u_val = norm.cdf(theta_val, loc=true_val, scale=4 * sigma)
                elif normal: # Non-angle, normal with Fisher
                    loc = true_val
                    scale = 4 * sigma
                    u_val = norm.cdf(theta_val, loc=loc, scale=scale)
                else: # Uniform distribution with Fisher uncertainties
                    min_linear = true_val - 4 * sigma
                    max_linear = true_val + 4 * sigma
                    u_val = (theta_val - min_linear) / (max_linear - min_linear)
            elif normal: # Normal distribution without Fisher uncertainties
                if is_angle:
                    delta = angle_diff(theta_val, true_val)
                    u_val = norm.cdf(delta, loc=0.0, scale=prange_val / 2.0)
                else:
                    loc = true_val
                    scale = prange_val / 2.0
                    u_val = norm.cdf(theta_val, loc=loc, scale=scale)
            else: # Uniform distribution without Fisher uncertainties
                if is_angle:
                    min_linear = true_val - prange_val / 2.0
                    max_linear = true_val + prange_val / 2.0
                    u_val = (angle_diff(theta_val, true_val) + prange_val / 2.0) / prange_val
                else:
                    min_linear = true_val - prange_val / 2.0
                    max_linear = true_val + prange_val / 2.0
                    u_val = (theta_val - min_linear) / (max_linear - min_linear)
            u[:, u_idx] = u_val

    return u


def runplot(self, res, event_name, path):
    """Create and save a Dynesty run plot.

    Parameters
    ----------
    res : dynesty.results.Results
        Results structure returned by ``dynesty`` containing the samples.
    event_name : str
        Name of the microlensing event used as a prefix for the saved file.
    path : str
        Directory where the plot will be written.

    Notes
    -----
    The figure is saved as ``<path>/posteriors/<event_name>_runplot.png``.
    """
    # This function seems okay as ``dyplot.runplot`` does not require explicit
    # labels or truths if they are already encoded in ``res``.
    if "run" in self.debug:
        print("debug Fit.runplot: event_name: ", event_name)
    fig, _ = dyplot.runplot(res)
    plt.title(event_name)
    fig.savefig(path + "posteriors/" + event_name + "_runplot.png")
    plt.close(fig)


def traceplot(self, res, event_name, path, truths):
    """Create and save a Dynesty trace plot.

    Parameters
    ----------
    res : dynesty.results.Results
        Results structure produced by ``dynesty``.
    event_name : str
        Name of the event used to annotate and save the figure.
    path : str
        Directory in which the plot will be saved.
    truths : dict
        Dictionary containing the true parameter values. Only the first
        ``self.ndim`` values are used.

    Notes
    -----
    The figure is saved as ``<path>/posteriors/<event_name>_traceplot.png``.
    """
    if "trace" in self.debug:
        print("debug Fit.traceplot: event_name: ", event_name)

    # Use labels and ndim stored in the Fit object (self)
    current_labels = self.labels
    # Slice truths according to the current number of dimensions
    current_truths = truths["params"][: self.ndim]

    fig, _ = dyplot.traceplot(
        res,
        truths=np.array(current_truths),
        truth_color="black",
        show_titles=True,
        trace_cmap="viridis",
        connect=True,
        connect_highlight=range(
            min(5, self.ndim)
        ),  # Highlight fewer if fewer params
        labels=current_labels,
    )  # Use dynamic labels

    if "trace" in self.debug:
        print("debug Fit.traceplot fig: built")
    plt.suptitle(
        event_name
    )  # Use suptitle for overall plot title with traceplot
    fig.savefig(path + "posteriors/" + event_name + "_traceplot.png")
    plt.close(fig)


def run_dynesty(self, event, event_name, ndim, path, truths, prange_linear, prange_log, normal, fisher_uncertainties_for_prior=None):
    """Run Dynesty for the given event.

    Parameters
    ----------
    event : Event
        The event to run Dynesty for.
    event_name : str
        The name of the event.
    ndim : int
        The number of dimensions of the parameter space.
    path : str
        The path to save the sampler to.
    truths : dict
        The true parameter values.
    prange_linear : array_like
        Linear prior widths for the current model parameters.
    prange_log : array_like
        Logarithmic prior widths for the current model parameters.
    normal : bool
        If True, use normal priors instead of uniform.
    fisher_uncertainties_for_prior : array_like or None, optional
        1-sigma Fisher uncertainties for each parameter.
    
    Returns
    -------
    sampler : dynesty.DynamicNestedSampler
        The sampler.
    """
    # Set current event for the likelihood
    self.current_event = event
    
    # Combine prange_linear and prange_log like in old code
    # For old code compatibility, create single prange array
    prange = np.concatenate([prange_log, prange_linear])
    
    sampler = dynesty.DynamicNestedSampler(
        self.lnprob,  # Use lnprob directly, not lnprob_transform
        self.prior_transform, 
        ndim, 
        nlive=100,  # Reduced from 200
        sample='rwalk', 
        bound='multi',
        logl_args=[event],  # Pass event as argument to lnprob
        ptform_args=[truths["params"][:ndim], prange_linear, prange_log],  # Pass args to prior_transform
        ptform_kwargs={'normal': normal, 'fisher_uncertainties': fisher_uncertainties_for_prior}  # Pass kwargs
    )
    
    sampler.run_nested(maxiter=500, print_progress=True)  # Removed checkpoint_file

    # Save the sampler as a pickle file
    with open(path+'posteriors/'+event_name+'_sampler.pkl', 'wb') as f:
        pickle.dump(sampler.results, f)

    res = sampler.results

    # print for logs
    print(f'Event {event_name} is done')
    print(res.summary())

    # Save plots
    self.corner_post(res.samples, event_name, path, truths)
    self.runplot(res, event_name, path)
    self.traceplot(res, event_name, path, truths)

    samples = res.samples
    np.save(path+'posteriors/'+event_name+'_post_samples.npy', samples)
    
    return sampler