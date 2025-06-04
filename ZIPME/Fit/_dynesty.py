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


def prior_transform(self, u, true, prange, normal=False):
    """Transform unit cube to the parameter space. Nested sampling has firm boundaries on the prior space."""

    if "pt" in self.debug:
        print(
            "debug: Fit.prior_transform: logs, logq, logrho, u0, alpha, t0, tE, piEE, piEN, i, sinphase, logperiod:"
        )
        print("                            ", u)

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
    logperiod_true = np.log10(true[11] * 1.0)
    true_array = np.array(
        [
            logs_true,
            logq_true,
            logrho_true,
            u0_true,
            alpha_true,
            t0_true,
            tE_true,
            piEE_true,
            piEN_true,
            i_true,
            phase_true,
            logperiod_true,
        ]
    )

    if "pt" in self.debug:
        print("        log true         ", true_array)

    min_array = true_array - prange
    max_array = true_array + prange

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

        x = normal(u, true_array, prange / 5.0, [min_array, max_array])

        if "pt" in self.debug:
            print("            normal           ", x)
            print("            min              ", min_array)
            print("            max              ", max_array)
            print("            true             ", true_array)

    else:
        x = (u - 0.5) * prange + true_array

    if u.ndim == 2:
        logs = x[:, 0]
        logq = x[:, 1]
        logrho = x[:, 2]
        u0 = x[:, 3]
        alpha = x[:, 4]
        t0 = x[:, 5]
        tE = x[:, 6]
        piEE = x[:, 7]
        piEN = x[:, 8]
        i = x[:, 9]
        phase = x[:, 10]
        logperiod = x[:, 11]
    else:
        (
            logs,
            logq,
            logrho,
            u0,
            alpha,
            t0,
            tE,
            piEE,
            piEN,
            i,
            phase,
            logperiod,
        ) = x

    s = 10.0**logs  # log uniform samples
    q = 10.0**logq
    rho = 10.0**logrho
    # print('debug: Fit.prior_transform: phase: ', phase)
    period = 10.0**logperiod

    if "pt" in self.debug:
        print(
            "debug Fit.prior_transform: s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period: "
        )
        print(
            "                  ",
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
        )
        print("       truths:    ", true)
    else:
        print("~", end="")

    if u.ndim == 2:
        x[:, 2] = rho
        x[:, 0] = s
        x[:, 1] = q
        x[:, 11] = period
    else:
        x[2] = rho
        x[0] = s
        x[1] = q
        x[11] = period

    return x


def runplot(self, res, event_name, path):
    if "run" in self.debug:
        print("debug Fit.runplot: event_name: ", event_name)
        print("debug Fit.runplot: path: ", path)

    fig, _ = dyplot.runplot(res)

    if "run" in self.debug:
        print("debug Fit.runplot: dyplot.runplot fig: built")

    plt.title(event_name)

    fig.savefig(path + "posteriors/" + event_name + "_runplot.png")

    if "run" in self.debug:
        print(
            "debug Fit.runplot: save path: ",
            path + "posteriors/" + event_name + "_runplot.png",
        )

    plt.close(fig)


def traceplot(self, res, event_name, path, truths):
    if "trace" in self.debug:
        print("debug Fit.traceplot: event_name: ", event_name)
        print("debug Fit.traceplot: path: ", path)
        print("debug Fit.traceplot: truths: ", truths)

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
    fig, _ = dyplot.traceplot(
        res,
        truths=np.array(truths["params"]),
        truth_color="black",
        show_titles=True,
        trace_cmap="viridis",
        connect=True,
        connect_highlight=range(5),
        labels=labels,
    )

    if "trace" in self.debug:
        print("debug Fit.traceplot: dyplot.traceplot fig: built")

    plt.title(event_name)

    fig.savefig(path + "posteriors/" + event_name + "_traceplot.png")

    if "trace" in self.debug:
        print(
            "debug Fit.traceplot: save path: ",
            path + "posteriors/" + event_name + "_traceplot.png",
        )

    plt.close(fig)
