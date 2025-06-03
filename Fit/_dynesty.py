# In Fit/_dynesty.py
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
# from dynesty import sampling as dysample # Not used in the functions you provided
# from dynesty import DynamicNestedSampler # Not used
# from dynesty import NestedSampler # Not used
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm # norm was missing for normal transform
import time # Not used here but was in original
import os # Not used here
import sys # Not used here
import numpy as np


def prior_transform(self, u, true_full, prange_linear, prange_log, normal=False):
    """Map unit-cube samples to physical parameters.

    Parameters
    ----------
    u : array_like
        Samples from the unit hypercube with shape ``(nwalkers, ndim)`` or
        ``(ndim,)``.
    true_full : array_like
        Reference parameter values for the complete 12-parameter model.
    prange_linear : array_like
        Linear prior widths for the current model parameters.
    prange_log : array_like
        Logarithmic prior widths for the current model parameters.
    normal : bool, optional
        If ``True``, draw from normal rather than uniform distributions.

    Returns
    -------
    ndarray
        Array of transformed parameters with the same shape as ``u``.
    """
    theta = np.zeros_like(u) # Output array, same shape as u
    
    # Use the labels stored in the Fit object to determine current parameter set
    current_labels = self.labels 
    
    # Define which parameter names are log-transformed
    log_param_names_base = ['s', 'q', 'rho']
    if self.LOM_enabled:
        log_param_names = log_param_names_base + ['period']
    else:
        log_param_names = log_param_names_base

    # Get indices for log and linear params within the *current* model (9 or 12 params)
    # These indices refer to positions within `u` and `theta`
    u_log_indices = [i for i, label in enumerate(current_labels) if label in log_param_names]
    u_linear_indices = [i for i, label in enumerate(current_labels) if label not in log_param_names]

    # Get indices for log and linear params within the *full 12-param truth array*
    full_labels_list = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
    true_log_indices = [full_labels_list.index(name) for name in log_param_names]
    
    # For linear params, we need to map current linear labels to their index in the full truth array
    # First, get the names of the current linear parameters
    current_linear_labels = [label for label in current_labels if label not in log_param_names]
    true_linear_indices = [full_labels_list.index(name) for name in current_linear_labels]

    # Slice the 'true_full' array (which is always 12 params)
    true_log_values = np.asarray(true_full)[true_log_indices]
    true_linear_values = np.asarray(true_full)[true_linear_indices]

    # --- Transform log parameters ---
    # prange_log is already correctly sized
    if u.ndim == 1: # Single sample
        for i, u_idx in enumerate(u_log_indices):
            true_val = true_log_values[i]
            prange_val = prange_log[i]
            if normal:
                loc = np.log10(true_val)
                scale = prange_val / 2.0 
                theta[u_idx] = 10**norm.ppf(u[u_idx], loc=loc, scale=scale)
            else:
                min_log = np.log10(true_val) - prange_val / 2.0
                max_log = np.log10(true_val) + prange_val / 2.0
                theta[u_idx] = 10**(min_log + (max_log - min_log) * u[u_idx])
    else: # Multiple samples (walkers)
        for i, u_idx in enumerate(u_log_indices):
            true_val = true_log_values[i]
            prange_val = prange_log[i]
            if normal:
                loc = np.log10(true_val)
                scale = prange_val / 2.0
                theta[:, u_idx] = 10**norm.ppf(u[:, u_idx], loc=loc, scale=scale)
            else:
                min_log = np.log10(true_val) - prange_val / 2.0
                max_log = np.log10(true_val) + prange_val / 2.0
                theta[:, u_idx] = 10**(min_log + (max_log - min_log) * u[:, u_idx])
                
    # --- Transform linear parameters ---
    # prange_linear is already correctly sized
    if u.ndim == 1:
        for i, u_idx in enumerate(u_linear_indices):
            true_val = true_linear_values[i]
            prange_val = prange_linear[i]
            if normal:
                loc = true_val
                scale = prange_val / 2.0
                theta[u_idx] = norm.ppf(u[u_idx], loc=loc, scale=scale)
            else:
                min_linear = true_val - prange_val / 2.0
                max_linear = true_val + prange_val / 2.0
                theta[u_idx] = min_linear + (max_linear - min_linear) * u[u_idx]
    else:
        for i, u_idx in enumerate(u_linear_indices):
            true_val = true_linear_values[i]
            prange_val = prange_linear[i]
            if normal:
                loc = true_val
                scale = prange_val / 2.0
                theta[:, u_idx] = norm.ppf(u[:, u_idx], loc=loc, scale=scale)
            else:
                min_linear = true_val - prange_val / 2.0
                max_linear = true_val + prange_val / 2.0
                theta[:, u_idx] = min_linear + (max_linear - min_linear) * u[:, u_idx]

    # Angle wrapping based on current labels
    param_to_theta_idx = {label: i for i, label in enumerate(current_labels)}

    if 'alpha' in param_to_theta_idx:
        alpha_idx = param_to_theta_idx['alpha']
        if u.ndim == 1: theta[alpha_idx] %= (2 * np.pi)
        else: theta[:, alpha_idx] %= (2 * np.pi)
        
    if self.LOM_enabled:
        if 'i' in param_to_theta_idx: # Inclination
            i_idx = param_to_theta_idx['i']
            # Decide if 'i' should be 0-pi or 0-2pi. Typically 0-pi.
            # If 0-pi, use something like: val = val % (2*np.pi); if val > np.pi: val -= np.pi
            if u.ndim == 1: theta[i_idx] %= (2 * np.pi) 
            else: theta[:, i_idx] %= (2 * np.pi)
        if 'phase' in param_to_theta_idx:
            phase_idx = param_to_theta_idx['phase']
            if u.ndim == 1: theta[phase_idx] %= (2 * np.pi)
            else: theta[:, phase_idx] %= (2 * np.pi)
            
    return theta
 

def runplot(self, res, event_name, path):
    # This function seems okay as dyplot.runplot doesn't require explicit labels or truths
    # if they are already in the 'res' object from dynesty.
    if 'run' in self.debug:
        print('debug Fit.runplot: event_name: ', event_name)
    fig, _ = dyplot.runplot(res)
    plt.title(event_name)
    fig.savefig(path+'posteriors/'+event_name+'_runplot.png')
    plt.close(fig)

def traceplot(self, res, event_name, path, truths):
    if 'trace' in self.debug:
        print('debug Fit.traceplot: event_name: ', event_name)

    # Use labels and ndim stored in the Fit object (self)
    current_labels = self.labels 
    # Slice truths according to the current number of dimensions
    current_truths = truths['params'][:self.ndim]

    fig, _ = dyplot.traceplot(res, 
                              truths=np.array(current_truths),
                              truth_color='black', 
                              show_titles=True,
                              trace_cmap='viridis', 
                              connect=True,
                              connect_highlight=range(min(5, self.ndim)), # Highlight fewer if fewer params 
                              labels=current_labels) # Use dynamic labels
        
    if 'trace' in self.debug:
        print('debug Fit.traceplot fig: built')
    plt.suptitle(event_name) # Use suptitle for overall plot title with traceplot
    fig.savefig(path+'posteriors/'+event_name+'_traceplot.png')
    plt.close(fig)
