from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty import sampling as dysample
from dynesty import DynamicNestedSampler
from dynesty import NestedSampler
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import time
import os
import sys
import numpy as np


'''
def prior_transform(self, u, true, prange, normal=False):
    """Transform unit cube to the parameter space. Nested sampling has firm boundaries on the prior space."""
        
    if 'pt' in self.debug:
        print('debug: Fit.prior_transform: logs, logq, logrho, u0, alpha, t0, tE, piEE, piEN, i, sinphase, logperiod:')
        print('                            ', u)

    logs_true = np.log10(true[0])
    logq_true = np.log10(true[1])
    logrho_true = np.log10(true[2])
    u0_true = true[3]
    alpha_true = true[4]
    t0_true = true[5]
    tE_true = true[6]
    piEE_true = true[7]
    piEN_true = true[8]
    i_true = true[9]
    phase_true = true[10]
    logperiod_true = np.log10(true[11]*1.0)
    true_array = np.array([logs_true, logq_true, logrho_true, u0_true, alpha_true, t0_true, tE_true, piEE_true, piEN_true, i_true, phase_true, logperiod_true])

    if 'pt' in self.debug:
        print('        log true         ', true_array)

    min_array = true_array - prange/2
    max_array = true_array + prange/2

    if normal:

        def normal(u, mu, sig, bounds):
            """Maps a uniform random variable u (between 0 and 1) to a truncated normal random value x,
            constrained between bounds[0] and bounds[1], with a mean of mu and standard deviation of sig.

            Parameters:
            u (float): Uniform random variable between 0 and 1.
            mu (float): Mean of the normal distribution.
            sig (float): Standard deviation of the normal distribution.
            bounds (tuple): Tuple containing the lower and upper bounds (bounds[0], bounds[1]).

            Returns:
            float: Truncated normal random value x constrained between bounds[0] and bounds[1].
            """

            # Calculate the lower and upper bounds in terms of the standard normal distribution
            a, b = (bounds[0] - mu) / sig, (bounds[1] - mu) / sig
                
            # Create a truncated normal distribution
            trunc_normal = truncnorm(a, b, loc=mu, scale=sig)
                
            # Map the uniform random variable to the truncated normal distribution
            if u.ndim == 2:
                x = np.zeros_like(u)
                for i in range(u.shape[0]):
                    x[i] = trunc_normal.ppf(u[i])
            else:
                x = trunc_normal.ppf(u)
                
            return x

        x = normal(u, true_array, prange/5.0, [min_array, max_array])

        if 'pt' in self.debug:
            print('            normal           ', x)
            print('            min              ', min_array)
            print('            max              ', max_array)
            print('            true             ', true_array)


    else:
        x = (u-0.5)*prange+true_array

    if u.ndim == 2:
        logs = x[:,0]
        logq = x[:,1]
        logrho = x[:,2]
        u0 = x[:,3]
        alpha = x[:,4]
        t0 = x[:,5]
        tE = x[:,6]
        piEE = x[:,7]
        piEN = x[:,8]
        i = x[:,9]
        phase = x[:,10]
        logperiod = x[:,11]
    else:
        logs, logq, logrho, u0, alpha, t0, tE, piEE, piEN, i, phase, logperiod = x

    s = 10.0**logs  # log uniform samples
    q = 10.0**logq
    rho = 10.0**logrho
    #print('debug: Fit.prior_transform: phase: ', phase)
    period = 10.0**logperiod

    if 'pt' in self.debug:
        print('debug Fit.prior_transform: s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period: ')
        print('                  ', s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period)
        print('       truths:    ', true)
    else:
        print('~', end='')
        
    if u.ndim == 2:
        x[:,2] = rho
        x[:,0] = s
        x[:,1] = q
        x[:,11] = period
    else:
        x[2] = rho
        x[0] = s
        x[1] = q
        x[11] = period

    return x #'''

# In Fit/_dynesty.py
import numpy as np # Make sure numpy is imported in this file

def prior_transform(self, u, true, prange_linear, prange_log, normal=False):
    """
    Transform unit cube to the parameter space.
    Handles linear and log-space parameters separately.
    """
    # THE FIX: Ensure the 'true' parameter is a NumPy array for fancy indexing
    true = np.asarray(true)

    # Define which parameter indices are linear vs. log
    # s, q, rho, period are log
    log_indices = [0, 1, 2, 11]
    # u0, alpha, t0, tE, piEE, piEN, i, phase are linear
    linear_indices = [3, 4, 5, 6, 7, 8, 9, 10]

    # --- Handle linear parameters ---
    true_linear = true[linear_indices]
    min_linear = true_linear - prange_linear / 2.0
    max_linear = true_linear + prange_linear / 2.0
    sig_linear = prange_linear / 5.0 # Or whatever factor you deem appropriate

    # --- Handle log parameters ---
    # Convert true values to log10 space
    true_log = np.log10(true[log_indices])
    min_log = true_log - prange_log / 2.0
    max_log = true_log + prange_log / 2.0
    sig_log = prange_log / 5.0 # Or whatever factor you deem appropriate

    # The full 'mu' and 'sigma' vectors for the truncated normal
    true_array = np.zeros(len(true))
    true_array[linear_indices] = true_linear
    true_array[log_indices] = true_log

    sig_array = np.zeros(len(true))
    sig_array[linear_indices] = sig_linear
    sig_array[log_indices] = sig_log

    min_array = np.zeros(len(true))
    min_array[linear_indices] = min_linear
    min_array[log_indices] = min_log

    max_array = np.zeros(len(true))
    max_array[linear_indices] = max_linear
    max_array[log_indices] = max_log

    if normal:
        # Calculate the bounds in terms of the standard normal distribution
        a, b = (min_array - true_array) / sig_array, (max_array - true_array) / sig_array

        # Create the truncated normal distribution
        trunc_normal = truncnorm(a, b, loc=true_array, scale=sig_array)

        # Map the uniform random variable to the truncated normal distribution
        if u.ndim == 2:
            x = np.zeros_like(u)
            for i in range(u.shape[0]):
                x[i] = trunc_normal.ppf(u[i])
        else:
            x = trunc_normal.ppf(u)
    else:
        # Simple uniform transformation
        full_prange = max_array - min_array
        x = (u - 0.5) * full_prange + true_array

    # Now, create the final physical parameter array
    final_params = np.zeros_like(x)
    if u.ndim == 2:
        # Get the linear-space values
        final_params[:, linear_indices] = x[:, linear_indices]
        # Convert log-space values back to linear and place them
        final_params[:, log_indices] = 10.0**x[:, log_indices]
    else:
        final_params[linear_indices] = x[linear_indices]
        final_params[log_indices] = 10.0**x[log_indices]

    return final_params
 
def runplot(self, res, event_name, path):
    if 'run' in self.debug:
        print('debug Fit.runplot: event_name: ', event_name)
        print('debug Fit.runplot: path: ', path)

    fig, _ = dyplot.runplot(res)

    if 'run' in self.debug:
        print('debug Fit.runplot: dyplot.runplot fig: built')

    plt.title(event_name)

    fig.savefig(path+'posteriors/'+event_name+'_runplot.png')

    if 'run' in self.debug:
        print('debug Fit.runplot: save path: ', path+'posteriors/'+event_name+'_runplot.png')    

    plt.close(fig)

def traceplot(self, res, event_name, path, truths):
    if 'trace' in self.debug:
        print('debug Fit.traceplot: event_name: ', event_name)
        print('debug Fit.traceplot: path: ', path)
        print('debug Fit.traceplot: truths: ', truths)

    labels = ['s', 'q', 'rho', 'u0', 'alpha', 't0', 'tE', 'piEE', 'piEN', 'i', 'phase', 'period']
    fig, _ = dyplot.traceplot(res, 
                              truths=np.array(truths['params']),
                              truth_color='black', 
                              show_titles=True,
                              trace_cmap='viridis', 
                              connect=True,
                              connect_highlight=range(5), 
                              labels=labels)
        
    if 'trace' in self.debug:
        print('debug Fit.traceplot: dyplot.traceplot fig: built')

    plt.title(event_name)

    fig.savefig(path+'posteriors/'+event_name+'_traceplot.png')

    if 'trace' in self.debug:
        print('debug Fit.traceplot: save path: ', path+'posteriors/'+event_name+'_traceplot.png')

    plt.close(fig)
