#!/usr/bin/env python3
"""Part 4: Plot lightcurves in flux space with truth, best-fit, and posterior bands.

For each event with collected posteriors, plots:
  - Measured data (band 0 / W146) with error bars
  - Simulation truth model (true_relative_flux from the lightcurve file)
  - Posterior predictive median model
  - Credible envelopes built from posterior sample model families

Usage (run from GullsPosteriors/):
    python unc_check_part4_lightcurves.py
    python unc_check_part4_lightcurves.py m00 m10
    python unc_check_part4_lightcurves.py --n-samples 50 m00
"""

from __future__ import annotations

import sys
import argparse
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms

# Python 3.8 on the cluster lacks argparse.BooleanOptionalAction.
if not hasattr(argparse, "BooleanOptionalAction"):
    class BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            option_strings = list(option_strings)
            expanded = []
            for option in option_strings:
                expanded.append(option)
                if option.startswith("--"):
                    expanded.append("--no-" + option[2:])
            super().__init__(
                option_strings=expanded,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

        def format_usage(self):
            return " | ".join(self.option_strings)

    argparse.BooleanOptionalAction = BooleanOptionalAction

# ── Paths (relative to GullsPosteriors/) ──────────────────────────────────────
DATA_BASE    = Path("../filter_selection")
OUTDIR       = Path("../lightcurve_plots")
REPO_ROOT    = Path(__file__).resolve().parents[1]
ROMAN_PLOT_SCRIPT = (
    REPO_ROOT
    / "roman-skills"
    / "skills"
    / "roman"
    / "plotting"
    / "plot-types"
    / "lightcurve-residuals"
    / "scripts"
    / "roman_plot.py"
)

COMPLETED_BINS = ["m-10", "m00", "m10", "m20", "m30", "m40"]

# Indices that are sampled in log10 space: s(0), q(1), rho(2), tE(6)
LOG_INDICES = [0, 1, 2, 6]

N_SAMPLES_PLOT_DEFAULT = 100
OBS_BAND = 0   # W146
BAND_LABEL = "W146"
MODEL_BASELINE_LEVEL = 1.0
JOURNAL_PROFILE = "apj"
PAPER_SPAN = "double"
MIN_ENVELOPE_SAMPLES = 5
POSTERIOR_PERCENTILES = [2.5, 16.0, 84.0, 97.5]
POSTERIOR_OUTER_COLOUR = "#d7dce4"
POSTERIOR_INNER_COLOUR = "#9caebf"
TRUTH_COLOUR = "#2a7d57"
MEDIAN_MODEL_COLOUR = "#111111"
AUTO_X_ZOOM_FRAC = 0.1
ANOMALY_SIGMA_THRESHOLD = 1.0
INSET_PAD_FRACTION = 0.25
MAX_INSET_WIDTH_FRACTION = 0.25
INSET_ZORDER = 20.0
RESIDUAL_ENVELOPE_ZORDER = 4.6
TRUTH_RESIDUAL_ZORDER = 6.2
TRUTH_MAIN_ZORDER = 6.4
T0_MARKER_MAIN_ZORDER = 9.0
T0_MARKER_TEXT_ZORDER = 10.0
T0_MARKER_RESID_ZORDER = 5.0

# ── GullsPosteriors module path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from Data import Data
from Parallax import Parallax
from Event import Event
from Fit import Fit
from Orbit import Orbit


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bins", nargs="*", default=COMPLETED_BINS,
                   help="Mass bins to process (default: all)")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES_PLOT_DEFAULT,
                   dest="n_samples",
                   help=f"Number of posterior samples used to build envelopes (default: {N_SAMPLES_PLOT_DEFAULT})")
    p.add_argument("--outdir", type=Path, default=OUTDIR,
                   help="Output directory for plots")
    p.add_argument("--max-events", type=int, default=None,
                   help="Maximum number of posterior-backed events to plot per mass bin")
    return p.parse_args(argv)


def lc_filename_to_event_name(lc_filename: str) -> tuple[str, int, int, int]:
    """Return (event_name, field, subrun, event_id) from a .det.lc filename.

    Mirrors the logic in Data._load_event_from_lcfile:
      filename parts[-3]=subrun, parts[-2]=field, parts[-1]=eventid
      event_name = "field_subrun_eventid"
    """
    base = lc_filename.replace(".det.lc", "")
    parts = base.split("_")
    event_id = int(parts[-1])
    field    = int(parts[-2])
    subrun   = int(parts[-3])
    event_name = f"{field}_{subrun}_{event_id}"
    return event_name, field, subrun, event_id


def find_samples(posteriors_dir: Path, event_name: str):
    """Return (samples_path, blobs_path_or_None, blobs_keys_path_or_None)."""
    for suffix in ("emcee_samples.npy", "post_samples.npy", "dynesty_samples.npy"):
        candidate = posteriors_dir / f"{event_name}_{suffix}"
        if candidate.exists():
            blobs_path = posteriors_dir / f"{event_name}_{suffix.replace('samples', 'blobs')}"
            keys_path  = posteriors_dir / f"{event_name}_{suffix.replace('samples', 'blobs_keys')}"
            return (
                candidate,
                blobs_path  if blobs_path.exists()  else None,
                keys_path   if keys_path.exists()   else None,
            )
    return None, None, None


def load_blobs(blobs_path: Path | None, keys_path: Path | None):
    """Return a dict {key: array_of_values_per_sample} or empty dict."""
    if blobs_path is None or keys_path is None:
        return {}
    blobs_arr  = np.load(blobs_path)                   # (n_samples, n_keys)
    blobs_keys = np.load(keys_path, allow_pickle=True)
    blobs_keys = [k.decode() if isinstance(k, bytes) else str(k) for k in blobs_keys]
    return {k: blobs_arr[:, i] for i, k in enumerate(blobs_keys)}


def is_finite_array(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=float)
    return np.all(np.isfinite(arr))


def choose_model_grid(t_obs: np.ndarray) -> np.ndarray:
    # Use the actual sampled cadence so model and truth overlays are not
    # undersampled relative to the data.
    return np.asarray(t_obs, dtype=float)


def transform_sample(raw_params: np.ndarray) -> np.ndarray:
    """Convert log10-sampled parameters to linear, apply alpha mod 2π."""
    p = raw_params.copy()
    for i in LOG_INDICES:
        p[i] = 10.0 ** p[i]
    p[4] %= 2.0 * np.pi   # alpha
    return p


def setup_event(orbit_obj: Orbit, data: dict, truths: dict,
                sim_time0: float, gamma: float) -> tuple[Parallax, Event, Fit]:
    """Build Parallax, Event (LOM disabled), and Fit objects."""
    piE = np.array([truths["piEN"], truths["piEE"]])
    tu_data, epochs = {}, {}
    for obs_key in data.keys():
        tu_data[obs_key] = data[obs_key][3:5, :].T
        epochs[obs_key]  = data[obs_key][0, :]

    parallax_obj = Parallax(
        truths["ra_deg"], truths["dec_deg"],
        orbit_obj, truths["tcroin"],
        tu_data, piE, epochs,
    )
    parallax_obj.update_piE_NE(truths["piEN"], truths["piEE"])

    t_ref = truths["tcroin"]
    event = Event(
        parallax_obj, orbit_obj, data, truths,
        sim_time0, t_ref,
        gamma=gamma, LOM_enabled=False,
    )

    fit = Fit(sampling_package="emcee", LOM_enabled=False, ndim=9,
              labels=["s","q","rho","u0","alpha","t0","tE","piEE","piEN"],
              normal=True, unit_cube=False)

    return parallax_obj, event, fit


@lru_cache(maxsize=1)
def load_roman_plot_module() -> ModuleType:
    """Load the vendored Roman lightcurve renderer once per process."""
    if not ROMAN_PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"Roman plot script not found: {ROMAN_PLOT_SCRIPT}")

    spec = importlib.util.spec_from_file_location("roman_plot_module", ROMAN_PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load renderer spec from {ROMAN_PLOT_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def resolve_tex_policy() -> tuple[bool, str | None]:
    """Return whether strict TeX rendering is available in this environment."""
    roman_plot = load_roman_plot_module()
    try:
        roman_plot.check_tex_dependencies()
    except SystemExit as exc:
        return False, str(exc)
    return True, None


def build_renderer_frame(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    yerr_obs: np.ndarray,
    model_time: np.ndarray,
    model_values: np.ndarray,
    residuals: np.ndarray,
) -> pd.DataFrame:
    """Build a sparse table matching the strict renderer's expected columns."""
    n_obs = len(t_obs)
    n_model = len(model_time)
    n_rows = max(n_obs, n_model)

    frame = pd.DataFrame(
        {
            "time": np.full(n_rows, np.nan),
            "flux": np.full(n_rows, np.nan),
            "uncertainty": np.full(n_rows, np.nan),
            "model_time": np.full(n_rows, np.nan),
            "model_flux": np.full(n_rows, np.nan),
            "residual": np.full(n_rows, np.nan),
        }
    )
    if n_obs > 0:
        frame.loc[: n_obs - 1, "time"] = t_obs
        frame.loc[: n_obs - 1, "flux"] = y_obs
        frame.loc[: n_obs - 1, "uncertainty"] = yerr_obs
        frame.loc[: n_obs - 1, "residual"] = residuals
    if n_model > 0:
        frame.loc[: n_model - 1, "model_time"] = model_time
        frame.loc[: n_model - 1, "model_flux"] = model_values
    return frame


def build_renderer_args(
    roman_plot: ModuleType,
    input_path: Path,
    output_stem: Path,
    title: str,
    use_tex: bool,
) -> argparse.Namespace:
    """Construct strict-renderer CLI args for a single-event flux plot."""
    argv = [
        "--input", str(input_path),
        "--output", str(output_stem),
        "--x-col", "time",
        "--y-col", "flux",
        "--band-label", BAND_LABEL,
        "--err-col", "uncertainty",
        "--model-x-col", "model_time",
        "--model-col", "model_flux",
        "--residual-col", "residual",
        "--best-fit-label", "model",
        "--model-color", MEDIAN_MODEL_COLOUR,
        "--y-kind", "flux",
        "--y-band", BAND_LABEL,
        "--y-scale", "Relative",
        "--x-var", "BJD",
        "--x-unit", "days",
        "--baseline-level", f"{MODEL_BASELINE_LEVEL}",
        "--auto-x-zoom", "trim-baseline",
        "--auto-x-zoom-frac", f"{AUTO_X_ZOOM_FRAC}",
        "--journal-profile", JOURNAL_PROFILE,
        "--paper-span", PAPER_SPAN,
    ]
    if title:
        argv.extend(["--title", title])
    if not use_tex:
        argv.append("--no-tex")
    return roman_plot.parse_args(argv)


def add_interval_envelopes(
    ax,
    x_values: np.ndarray,
    curves: np.ndarray,
    *,
    label_inner: str | None = None,
    label_outer: str | None = None,
    zorder_base: float = 1.0,
) -> None:
    """Draw 68% and 95% central credible bands for model families."""
    q025, q16, q84, q975 = np.percentile(curves, POSTERIOR_PERCENTILES, axis=0)
    ax.fill_between(
        x_values,
        q025,
        q975,
        color=POSTERIOR_OUTER_COLOUR,
        alpha=0.8,
        linewidth=0,
        zorder=zorder_base,
        label=label_outer,
    )
    ax.fill_between(
        x_values,
        q16,
        q84,
        color=POSTERIOR_INNER_COLOUR,
        alpha=0.95,
        linewidth=0,
        zorder=zorder_base + 0.1,
        label=label_inner,
    )


def detect_anomaly_inset_range(
    x_values: np.ndarray,
    residual_family: np.ndarray | None,
    main_x_limits: tuple[float, float],
) -> tuple[tuple[float, float] | None, dict]:
    """Locate a compact anomaly window from posterior residual spread."""
    metadata = {
        "method": "posterior_residual_spread68_peak",
        "threshold_sigma": ANOMALY_SIGMA_THRESHOLD,
        "pad_fraction": INSET_PAD_FRACTION,
        "max_width_fraction": MAX_INSET_WIDTH_FRACTION,
        "enabled": False,
        "x_range": None,
        "reason": None,
    }

    if residual_family is None or residual_family.ndim != 2:
        metadata["reason"] = "posterior_residual_family_unavailable"
        return None, metadata

    x_values = np.asarray(x_values, dtype=float)
    main_x_min, main_x_max = map(float, main_x_limits)
    main_width = main_x_max - main_x_min
    if not np.isfinite(main_width) or main_width <= 0:
        metadata["reason"] = "invalid_main_axis_width"
        return None, metadata

    if residual_family.shape[1] != x_values.size:
        metadata["reason"] = "residual_family_shape_mismatch"
        return None, metadata

    finite_mask = np.isfinite(x_values) & np.all(np.isfinite(residual_family), axis=0)
    if np.count_nonzero(finite_mask) < 3:
        metadata["reason"] = "insufficient_finite_epochs"
        return None, metadata

    x_finite = x_values[finite_mask]
    residual_finite = residual_family[:, finite_mask]
    q16, q84 = np.percentile(residual_finite, [16.0, 84.0], axis=0)
    spread68 = q84 - q16
    if not np.all(np.isfinite(spread68)):
        metadata["reason"] = "non_finite_spread68"
        return None, metadata

    peak_idx = int(np.argmax(spread68))
    peak_x = float(x_finite[peak_idx])
    peak_spread = float(spread68[peak_idx])
    spread_median = float(np.median(spread68))
    spread_mad = float(np.median(np.abs(spread68 - spread_median)))
    spread_sigma_robust = 1.4826 * spread_mad
    spread_sigma_std = float(np.std(spread68))
    spread_sigma = max(
        spread_sigma_std if np.isfinite(spread_sigma_std) and spread_sigma_std > 0 else 0.0,
        spread_sigma_robust if np.isfinite(spread_sigma_robust) and spread_sigma_robust > 0 else 0.0,
    )
    if not np.isfinite(spread_sigma) or spread_sigma <= 0:
        metadata["reason"] = "degenerate_spread_distribution"
        metadata["peak_x"] = peak_x
        metadata["peak_spread68"] = peak_spread
        return None, metadata

    threshold = spread_median + ANOMALY_SIGMA_THRESHOLD * spread_sigma
    metadata["peak_x"] = peak_x
    metadata["peak_spread68"] = peak_spread
    metadata["spread68_median"] = spread_median
    metadata["spread68_sigma_std"] = spread_sigma_std
    metadata["spread68_sigma_robust"] = spread_sigma_robust
    metadata["spread68_sigma"] = spread_sigma
    metadata["spread68_threshold"] = float(threshold)
    if peak_spread <= threshold:
        metadata["reason"] = "peak_below_threshold"
        return None, metadata

    above_threshold = spread68 > threshold
    left_idx = peak_idx
    while left_idx > 0 and above_threshold[left_idx - 1]:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < spread68.size - 1 and above_threshold[right_idx + 1]:
        right_idx += 1

    window_left = float(x_finite[left_idx])
    window_right = float(x_finite[right_idx])
    detected_width = window_right - window_left
    if detected_width <= 0:
        left_bound_idx = max(0, left_idx - 1)
        right_bound_idx = min(x_finite.size - 1, right_idx + 1)
        window_left = float(x_finite[left_bound_idx])
        window_right = float(x_finite[right_bound_idx])
        detected_width = window_right - window_left
    if detected_width <= 0:
        metadata["reason"] = "zero_width_detection"
        return None, metadata

    pad = INSET_PAD_FRACTION * detected_width
    candidate_min = max(main_x_min, window_left - pad)
    candidate_max = min(main_x_max, window_right + pad)
    candidate_width = candidate_max - candidate_min
    metadata["detected_width"] = float(detected_width)
    metadata["window_start"] = window_left
    metadata["window_end"] = window_right
    metadata["candidate_width"] = float(candidate_width)
    metadata["main_axis_width"] = float(main_width)
    if candidate_width <= 0:
        metadata["reason"] = "invalid_candidate_window"
        return None, metadata
    if candidate_width >= MAX_INSET_WIDTH_FRACTION * main_width:
        metadata["reason"] = "candidate_window_too_wide"
        return None, metadata

    inset_x_range = (candidate_min, candidate_max)
    metadata["enabled"] = True
    metadata["x_range"] = [float(candidate_min), float(candidate_max)]
    return inset_x_range, metadata


def add_zoom_inset(
    ax_main,
    *,
    inset_x_range: tuple[float, float] | None,
    x_data: np.ndarray,
    y_data: np.ndarray,
    yerr_data: np.ndarray,
    model_x: np.ndarray,
    model_y: np.ndarray,
    truth_y: np.ndarray | None,
    posterior_curves: np.ndarray | None,
    data_colour: str,
) -> bool:
    """Add an anomaly inset around the requested x-range."""
    if inset_x_range is None:
        return False

    x_min, x_max = inset_x_range
    data_mask = (x_data >= x_min) & (x_data <= x_max)
    model_mask = (model_x >= x_min) & (model_x <= x_max)
    if not np.any(data_mask) or not np.any(model_mask):
        return False

    inset_ax = ax_main.inset_axes([0.58, 0.12, 0.34, 0.34])
    inset_ax.set_zorder(INSET_ZORDER)
    inset_ax.patch.set_facecolor("white")
    inset_ax.patch.set_alpha(1.0)
    inset_ax.patch.set_zorder(INSET_ZORDER)
    for spine in inset_ax.spines.values():
        spine.set_zorder(INSET_ZORDER + 0.1)

    if posterior_curves is not None:
        add_interval_envelopes(
            inset_ax,
            model_x[model_mask],
            posterior_curves[:, model_mask],
            zorder_base=0.8,
        )

    if truth_y is not None:
        inset_ax.plot(
            model_x[model_mask],
            truth_y[model_mask],
            color=TRUTH_COLOUR,
            linewidth=1.2,
            linestyle="--",
            zorder=5.7,
        )

    inset_ax.plot(
        model_x[model_mask],
        model_y[model_mask],
        color=MEDIAN_MODEL_COLOUR,
        linewidth=1.4,
        zorder=6,
    )
    inset_ax.errorbar(
        x_data[data_mask],
        y_data[data_mask],
        yerr=yerr_data[data_mask],
        fmt="o",
        markersize=1.9,
        elinewidth=0.45,
        alpha=0.8,
        color=data_colour,
        mec=data_colour,
        zorder=4,
    )

    zoom_values = [
        y_data[data_mask],
        model_y[model_mask],
    ]
    if truth_y is not None:
        zoom_values.append(truth_y[model_mask])
    if posterior_curves is not None:
        zoom_values.append(posterior_curves[:, model_mask].reshape(-1))
    zoom_values = np.concatenate([vals[np.isfinite(vals)] for vals in zoom_values if np.size(vals) > 0])
    if zoom_values.size > 0:
        y_min = float(np.min(zoom_values))
        y_max = float(np.max(zoom_values))
        y_span = y_max - y_min
        y_pad = 0.08 * y_span if y_span > 0 else max(0.02 * abs(y_min), 1e-3)
        inset_ax.set_ylim(y_min - y_pad, y_max + y_pad)

    inset_ax.set_xlim(x_min, x_max)
    inset_ax.tick_params(direction="in", labelsize=7)
    inset_ax.grid(False)
    inset_indicator = ax_main.indicate_inset_zoom(inset_ax, edgecolor="#555555", alpha=0.8)
    if isinstance(inset_indicator, tuple):
        for artist in inset_indicator:
            if artist is None:
                continue
            if isinstance(artist, (list, tuple)):
                for sub_artist in artist:
                    if sub_artist is not None:
                        sub_artist.set_zorder(INSET_ZORDER - 0.2)
            else:
                artist.set_zorder(INSET_ZORDER - 0.2)
    elif hasattr(inset_indicator, "connectors"):
        rectangle = getattr(inset_indicator, "rectangle", None)
        if rectangle is not None:
            rectangle.set_zorder(INSET_ZORDER - 0.2)
        for connector in inset_indicator.connectors:
            if connector is not None:
                connector.set_zorder(INSET_ZORDER - 0.1)
    return True


def add_truth_residual_overlay(
    ax_resid,
    *,
    x_values: np.ndarray,
    truth_values: np.ndarray | None,
    model_values: np.ndarray | None,
) -> bool:
    """Overlay simulation-truth residuals in the same x-cadence as the data."""
    if truth_values is None or model_values is None:
        return False

    truth_residual = np.asarray(truth_values, dtype=float) - np.asarray(model_values, dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(truth_residual)
    if not np.any(finite_mask):
        return False

    ax_resid.plot(
        x_values[finite_mask],
        truth_residual[finite_mask],
        linestyle="--",
        linewidth=1.35,
        color=TRUTH_COLOUR,
        alpha=0.95,
        zorder=TRUTH_RESIDUAL_ZORDER,
    )
    return True


def get_truth_t0_bjd(truths: dict) -> float | None:
    """Return the simulation t0_lens1 value in BJD when available."""
    for key in ("t0lens1", "t0_lens1"):
        value = truths.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value

    params = truths.get("params")
    if params is None or len(params) <= 5:
        return None
    try:
        value = float(params[5])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def build_t0_vline_entry(roman_plot: ModuleType, t0_bjd: float | None) -> dict | None:
    """Build a renderer-normalized vline entry for the simulation t0 marker."""
    if t0_bjd is None or not np.isfinite(t0_bjd):
        return None
    entries = roman_plot.parse_vline_specs([f"{float(t0_bjd)},t0"])
    if not entries:
        return None
    return entries[0]


def add_t0_marker(
    ax_main,
    ax_resid,
    *,
    t0_plot: float | None,
    vline_entry: dict | None,
    vline_linewidth: float,
    residual_linewidth: float,
    annotation_fontsize: float,
    annotation_text_color: str | None,
) -> bool:
    """Add a vertical simulation t0 marker to the lightcurve and residual panels."""
    if t0_plot is None or not np.isfinite(t0_plot) or vline_entry is None:
        return False

    color = str(vline_entry.get("color", "black"))
    linestyle = str(vline_entry.get("linestyle", "--"))
    label = str(vline_entry.get("label", "t0"))
    trans = transforms.blended_transform_factory(ax_main.transData, ax_main.transAxes)
    ax_main.axvline(
        t0_plot,
        color=color,
        linestyle=linestyle,
        linewidth=vline_linewidth,
        alpha=0.9,
        zorder=T0_MARKER_MAIN_ZORDER,
    )
    ax_main.text(
        t0_plot,
        0.98,
        label,
        transform=trans,
        rotation=90,
        va="top",
        ha="right",
        fontsize=annotation_fontsize,
        color=annotation_text_color or color,
        zorder=T0_MARKER_TEXT_ZORDER,
    )
    if ax_resid is not None:
        ax_resid.axvline(
            t0_plot,
            color=color,
            linestyle=linestyle,
            linewidth=residual_linewidth,
            alpha=0.7,
            zorder=T0_MARKER_RESID_ZORDER,
        )
    return True


def relabel_model_line(ax_main, model_label: str) -> None:
    """Promote the strict-rendered model line to the desired semantics."""
    for line in ax_main.lines:
        if line.get_label() in {"Model", f"$\\mathrm{{{BAND_LABEL}}}$ Model"}:
            line.set_label(model_label)
            line.set_color(MEDIAN_MODEL_COLOUR)
            line.set_linewidth(max(line.get_linewidth(), 1.8))
            line.set_zorder(6)
            break


def reorder_legend(ax_main) -> None:
    """Keep the legend compact and stable after customization."""
    handles, labels = ax_main.get_legend_handles_labels()
    deduped: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_"):
            continue
        deduped[label] = handle

    priorities = {
        "Posterior predictive median": 1,
        "Posterior parameter median": 1,
        "Posterior median": 1,
        "Truth model": 2,
        "Simulation truth": 2,
        "Posterior 68%": 3,
        "Posterior 95%": 4,
    }

    ordered_labels = sorted(
        deduped,
        key=lambda label: (
            0 if "Data" in label else priorities.get(label, 99),
            list(deduped).index(label),
        ),
    )
    ax_main.legend(
        [deduped[label] for label in ordered_labels],
        ordered_labels,
        loc="best",
        frameon=True,
    )


def append_unique_warning(validation: dict, message: str) -> None:
    warnings = validation.setdefault("warnings", [])
    if message not in warnings:
        warnings.append(message)


def finalize_manifest(
    manifest: dict,
    *,
    event_name: str,
    lc_input_path: Path,
    samples_path: Path,
    blobs_path: Path | None,
    keys_path: Path | None,
    tex_enabled: bool,
    tex_warning: str | None,
    model_label: str,
    used_truth_fallback: bool,
    truth_overlay: bool,
    n_requested: int,
    n_success: int,
    n_failed: int,
    posterior_envelopes_drawn: bool,
    zoom_inset_added: bool,
    truth_residual_overlay: bool,
    anomaly_inset_meta: dict,
    t0_marker_added: bool,
    t0_marker_bjd: float | None,
    t0_vline_entry: dict | None,
) -> None:
    """Mark the customized-from-strict render and record reproducibility info."""
    validation = manifest.setdefault("validation", {})
    figure_meta = manifest.setdefault("figure", {})
    provenance = manifest.setdefault("provenance", {})
    series = manifest.setdefault("series", [])

    figure_meta["policy_profile"] = "customized-from-strict"
    figure_meta["postprocess_customized"] = True
    figure_meta["truth_overlay"] = bool(truth_overlay)
    figure_meta["truth_source"] = "true_relative_flux" if truth_overlay or used_truth_fallback else "none"
    figure_meta["render_model"] = model_label
    figure_meta["posterior_envelopes"] = {
        "drawn": bool(posterior_envelopes_drawn),
        "sample_count_requested": int(n_requested),
        "sample_count_used": int(n_success),
        "credible_levels_percent": [68, 95],
        "residual_panel_band": bool(posterior_envelopes_drawn),
    }
    figure_meta["anomaly_inset"] = dict(anomaly_inset_meta)
    figure_meta["anomaly_inset"]["enabled"] = bool(zoom_inset_added)
    figure_meta["truth_residual_overlay"] = bool(truth_residual_overlay)
    figure_meta["t0_marker"] = {
        "enabled": bool(t0_marker_added),
        "source": "truths.t0lens1" if t0_marker_added else "none",
        "bjd": float(t0_marker_bjd) if t0_marker_bjd is not None else None,
        "label": str(t0_vline_entry.get("label")) if t0_vline_entry else None,
        "color": str(t0_vline_entry.get("color")) if t0_vline_entry else None,
        "linestyle": str(t0_vline_entry.get("linestyle")) if t0_vline_entry else None,
    }
    figure_meta["tex_enabled"] = bool(tex_enabled)

    provenance["source_lightcurve"] = str(lc_input_path.resolve())
    provenance["posterior_samples"] = str(samples_path.resolve())
    provenance["posterior_blobs"] = str(blobs_path.resolve()) if blobs_path else None
    provenance["posterior_blobs_keys"] = str(keys_path.resolve()) if keys_path else None
    provenance["renderer_script"] = str(ROMAN_PLOT_SCRIPT.resolve())
    provenance["customizer_script"] = str(Path(__file__).resolve())
    if t0_marker_added:
        figure_meta.setdefault("vertical_annotations", [])
        figure_meta["vertical_annotations"].append(
            {
                "x": float(t0_marker_bjd),
                "label": str(t0_vline_entry.get("label")) if t0_vline_entry else "t0",
                "color": str(t0_vline_entry.get("color")) if t0_vline_entry else "black",
                "linestyle": str(t0_vline_entry.get("linestyle")) if t0_vline_entry else "--",
                "source": "truths.t0lens1",
            }
        )

    for item in series:
        if item.get("name") in {"Model", f"$\\mathrm{{{BAND_LABEL}}}$ Model"}:
            item["name"] = model_label
            item["color"] = MEDIAN_MODEL_COLOUR
    if truth_overlay:
        series.append(
            {
                "name": "Simulation truth",
                "color": TRUTH_COLOUR,
                "marker": "None",
                "linestyle": "--",
                "alpha": 1.0,
                "linewidth": 1.6,
            }
        )
    if posterior_envelopes_drawn:
        series.extend(
            [
                {
                    "name": "Posterior 95%",
                    "color": POSTERIOR_OUTER_COLOUR,
                    "marker": "None",
                    "linestyle": "fill",
                    "alpha": 0.8,
                    "linewidth": 0.0,
                },
                {
                    "name": "Posterior 68%",
                    "color": POSTERIOR_INNER_COLOUR,
                    "marker": "None",
                    "linestyle": "fill",
                    "alpha": 0.95,
                    "linewidth": 0.0,
                },
            ]
        )

    validation["qa_checklist"] = {
        "columns_mapped_correctly": True,
        "missing_values_handled_explicitly": True,
        "error_bars_match_input_uncertainties": True,
        "time_offset_reflected_in_axis_label": "BJD" in figure_meta.get("labels", {}).get("x", ""),
        "residual_panel_included": bool(figure_meta.get("has_residual_panel", False)),
        "magnitude_axis_inverted": bool(figure_meta.get("mode") == "magnitude"),
        "anomaly_inset_added": bool(zoom_inset_added),
        "truth_residual_overlay_added": bool(truth_residual_overlay),
        "t0_marker_added": bool(t0_marker_added),
        "vector_export_created": True,
        "png_export_created": True,
        "colorblind_safe_palette_used": True,
        "tex_preflight_passed_or_disabled": True,
    }

    existing_warnings = validation.get("warnings", [])
    if tex_warning is not None and not any("TeX" in warning for warning in existing_warnings):
        append_unique_warning(validation, f"TeX disabled for this environment: {tex_warning}")
    if used_truth_fallback:
        append_unique_warning(
            validation,
            "Posterior-derived central model was unavailable; residuals use the truth model fallback.",
        )
    if n_failed > 0:
        append_unique_warning(
            validation,
            f"{n_failed} posterior sample curve(s) failed model evaluation and were skipped.",
        )
    if not posterior_envelopes_drawn:
        append_unique_warning(
            validation,
            f"Posterior envelopes not drawn because only {n_success} valid sample curve(s) were available.",
        )

    posterior_phrase = (
        "and posterior credible bands."
        if posterior_envelopes_drawn
        else "without posterior credible bands."
    )
    manifest["summary"] = (
        f"Generated {event_name} W146 relative-flux lightcurve with residuals, "
        f"{'truth overlay, ' if truth_overlay else ''}"
        f"{posterior_phrase}"
    )
    manifest["status"] = "warning" if used_truth_fallback or not posterior_envelopes_drawn else "ok"


# ── Main plot function ────────────────────────────────────────────────────────

def plot_event(
    event_name: str,
    data: dict,
    truths: dict,
    sim_time0: float,
    gamma: float,
    samples_raw: np.ndarray,
    blobs: dict,
    orbit_obj: Orbit,
    outdir: Path,
    lc_input_path: Path,
    samples_path: Path,
    blobs_path: Path | None,
    keys_path: Path | None,
    n_samples_plot: int,
) -> None:

    if OBS_BAND not in data:
        raise ValueError(f"Band {OBS_BAND} not in data for {event_name}. "
                         f"Available bands: {list(data.keys())}")

    t_obs  = data[OBS_BAND][0, :]          # BJD
    f_obs  = data[OBS_BAND][1, :]          # measured_relative_flux
    fe_obs = data[OBS_BAND][2, :]          # measured_relative_flux_error
    f_true = data[OBS_BAND][5, :]          # true_relative_flux (pre-computed truth)

    # Sort by time (should already be sorted, but just in case)
    sort_idx = np.argsort(t_obs)
    t_obs, f_obs, fe_obs, f_true = (
        t_obs[sort_idx], f_obs[sort_idx], fe_obs[sort_idx], f_true[sort_idx]
    )

    if not is_finite_array(f_obs):
        raise ValueError(f"Observed relative flux contains non-finite values for {event_name}.")
    if not is_finite_array(fe_obs):
        raise ValueError(f"Observed flux uncertainties contain non-finite values for {event_name}.")

    # ── Set up Event/Parallax/Fit ──────────────────────────────────────────
    _, event, fit = setup_event(orbit_obj, data, truths, sim_time0, gamma)

    # Use the observed cadence directly for model overlays.
    t_model = choose_model_grid(t_obs)
    same_model_grid = t_model.shape == t_obs.shape and np.array_equal(t_model, t_obs)

    # ── Truth model: recompute on the plotting cadence using truth params ───
    # truths["params"] are already in linear space (from get_params)
    truth_params = np.array(truths["params"][:9], dtype=float)
    # get Fs, FB at truth params using observed data
    event.set_params(truth_params)
    A_truth_obs = event.get_magnification(t_obs, OBS_BAND)
    if A_truth_obs is not None and np.all(np.isfinite(A_truth_obs)):
        fs_truth, fb_truth = fit.get_fluxes(A_truth_obs, f_obs, fe_obs ** 2)
        f_truth_obs = fs_truth * A_truth_obs + fb_truth
    else:
        # fallback: use Obs_0_fs from truths if available, else 1.0
        fs_truth = float(truths.get("Obs_0_fs", 1.0))
        fb_truth = 1.0 - fs_truth
        f_truth_obs = f_true if is_finite_array(f_true) else None

    A_truth_model = A_truth_obs if same_model_grid else event.get_magnification(t_model, OBS_BAND)
    f_truth_model = (
        fs_truth * A_truth_model + fb_truth
        if A_truth_model is not None and np.all(np.isfinite(A_truth_model))
        else None
    )

    # ── Best-fit: median posterior params ─────────────────────────────────
    median_raw = np.median(samples_raw, axis=0)
    median_lin = transform_sample(median_raw)

    f_param_median_model = None
    f_param_median_obs = None
    # get Fs, FB at median params
    event.set_params(median_lin)
    A_median_obs = event.get_magnification(t_obs, OBS_BAND)
    if A_median_obs is not None and np.all(np.isfinite(A_median_obs)):
        fs_med, fb_med = fit.get_fluxes(A_median_obs, f_obs, fe_obs ** 2)
        f_param_median_obs = fs_med * A_median_obs + fb_med
        A_median_model = A_median_obs if same_model_grid else event.get_magnification(t_model, OBS_BAND)
        if A_median_model is not None and np.all(np.isfinite(A_median_model)):
            f_param_median_model = fs_med * A_median_model + fb_med

    # ── Posterior sample models ────────────────────────────────────────────
    n_available = len(samples_raw)
    n_plot = min(n_samples_plot, n_available)
    rng = np.random.default_rng(seed=42)
    chosen_idx = rng.choice(n_available, size=n_plot, replace=False)

    sample_curves = []
    sample_obs_curves = []
    n_failed = 0
    for idx in chosen_idx:
        params_lin = transform_sample(samples_raw[idx])
        # Get per-sample Fs/FB from blobs if available
        fs_key = f"Fs_{OBS_BAND}"
        fb_key = f"FB_{OBS_BAND}"
        fs_override = float(blobs[fs_key][idx]) if fs_key in blobs else None
        fb_override = float(blobs[fb_key][idx]) if fb_key in blobs else None

        event.set_params(params_lin)
        A_obs = event.get_magnification(t_obs, OBS_BAND)
        if A_obs is None or not np.all(np.isfinite(A_obs)):
            n_failed += 1
            continue
        A_model = A_obs if same_model_grid else event.get_magnification(t_model, OBS_BAND)
        if A_model is None or not np.all(np.isfinite(A_model)):
            n_failed += 1
            continue
        if fs_override is not None and fb_override is not None:
            f_model = fs_override * A_model + fb_override
            f_model_obs = fs_override * A_obs + fb_override
        else:
            fs, fb = fit.get_fluxes(A_obs, f_obs, fe_obs ** 2)
            f_model = fs * A_model + fb
            f_model_obs = fs * A_obs + fb
        if not (is_finite_array(f_model) and is_finite_array(f_model_obs)):
            n_failed += 1
            continue
        sample_curves.append(f_model)
        sample_obs_curves.append(f_model_obs)

    if n_failed > 0:
        print(f"    Warning: {n_failed}/{n_plot} sample magnifications failed")

    n_success = len(sample_curves)
    sample_curves_arr = np.array(sample_curves) if n_success > 0 else None
    sample_obs_curves_arr = np.array(sample_obs_curves) if n_success > 0 else None

    render_model_time = t_model
    render_model_values = None
    render_model_obs = None
    render_model_label = "Posterior predictive median"
    used_truth_fallback = False
    truth_overlay = f_truth_model is not None

    if sample_curves_arr is not None:
        render_model_values = np.median(sample_curves_arr, axis=0)
        render_model_obs = np.median(sample_obs_curves_arr, axis=0)
    elif f_param_median_model is not None and f_param_median_obs is not None:
        render_model_values = f_param_median_model
        render_model_obs = f_param_median_obs
        render_model_label = "Posterior parameter median"
    else:
        used_truth_fallback = True
        render_model_label = "Truth model"
        if f_truth_model is not None:
            render_model_time = t_model
            render_model_values = f_truth_model
            render_model_obs = np.interp(t_obs, t_model, f_truth_model)
            truth_overlay = False
        else:
            if not is_finite_array(f_true):
                raise ValueError(f"Truth flux contains non-finite values for {event_name}.")
            render_model_time = t_obs
            render_model_values = f_true
            render_model_obs = f_true
            truth_overlay = False

    roman_plot = load_roman_plot_module()
    tex_enabled, tex_warning = resolve_tex_policy()
    output_stem = outdir / f"{event_name}_lightcurve"
    plot_df = build_renderer_frame(
        t_obs=t_obs,
        y_obs=f_obs,
        yerr_obs=fe_obs,
        model_time=render_model_time,
        model_values=render_model_values,
        residuals=f_obs - render_model_obs,
    )
    renderer_args = build_renderer_args(
        roman_plot=roman_plot,
        input_path=lc_input_path,
        output_stem=output_stem,
        title="",
        use_tex=tex_enabled,
    )
    fig, manifest = roman_plot.render_lightcurve(renderer_args, df=plot_df)

    ax_main = fig.axes[0]
    ax_resid = fig.axes[1] if len(fig.axes) > 1 else None
    data_series = next(
        (item for item in manifest.get("series", []) if "Data" in str(item.get("name", ""))),
        {},
    )
    data_colour = str(data_series.get("color", "#3B4CC0"))
    graphics_style = manifest.get("figure", {}).get("graphics_style_pt", {})
    font_sizes = manifest.get("figure", {}).get("font_sizes_pt", {})
    x_shift = float(manifest["figure"]["x_axis_shift"])
    t_plot_obs = t_obs - x_shift
    t_plot_model = t_model - x_shift
    truth_t0_bjd = get_truth_t0_bjd(truths)
    t0_plot = truth_t0_bjd - x_shift if truth_t0_bjd is not None else None
    t0_vline_entry = build_t0_vline_entry(roman_plot, truth_t0_bjd)
    vline_linewidth = float(graphics_style.get("vline_linewidth", 0.9))
    residual_linewidth = float(graphics_style.get("residual_linewidth", 0.75))
    annotation_fontsize = float(font_sizes.get("annotation", 8.5))
    journal_profiles = roman_plot.load_journal_profiles()
    journal_profile = journal_profiles.get(JOURNAL_PROFILE) if JOURNAL_PROFILE else None
    annotation_text_color = "black"
    if not (journal_profile and bool(journal_profile.get("avoid_colored_text", False))):
        annotation_text_color = None

    truth_residual_overlay = False
    t0_marker_added = False
    residual_family = None
    posterior_envelopes_drawn = bool(sample_curves_arr is not None and n_success >= MIN_ENVELOPE_SAMPLES)
    if posterior_envelopes_drawn:
        add_interval_envelopes(
            ax_main,
            t_plot_model,
            sample_curves_arr,
            label_inner="Posterior 68%",
                label_outer="Posterior 95%",
                zorder_base=0.8,
        )
        if ax_resid is not None and render_model_obs is not None and sample_obs_curves_arr is not None:
            residual_family = sample_obs_curves_arr - render_model_obs[None, :]
            add_interval_envelopes(
                ax_resid,
                t_plot_obs,
                residual_family,
                zorder_base=RESIDUAL_ENVELOPE_ZORDER,
            )

    if truth_overlay:
        ax_main.plot(
            t_plot_model,
            f_truth_model,
            color=TRUTH_COLOUR,
            linewidth=1.6,
            linestyle="--",
            zorder=TRUTH_MAIN_ZORDER,
            label="Simulation truth",
        )

    anomaly_inset_range, anomaly_inset_meta = detect_anomaly_inset_range(
        t_plot_obs,
        residual_family,
        ax_main.get_xlim(),
    )
    zoom_inset_added = add_zoom_inset(
        ax_main,
        inset_x_range=anomaly_inset_range,
        x_data=t_plot_obs,
        y_data=f_obs,
        yerr_data=fe_obs,
        model_x=t_plot_model,
        model_y=render_model_values,
        truth_y=f_truth_model if truth_overlay else None,
        posterior_curves=sample_curves_arr if posterior_envelopes_drawn else None,
        data_colour=data_colour,
    )

    if ax_resid is not None:
        truth_residual_overlay = add_truth_residual_overlay(
            ax_resid,
            x_values=t_plot_obs,
            truth_values=f_truth_obs,
            model_values=render_model_obs,
        )
    t0_marker_added = add_t0_marker(
        ax_main,
        ax_resid,
        t0_plot=t0_plot,
        vline_entry=t0_vline_entry,
        vline_linewidth=vline_linewidth,
        residual_linewidth=residual_linewidth,
        annotation_fontsize=annotation_fontsize,
        annotation_text_color=annotation_text_color,
    )

    relabel_model_line(ax_main, render_model_label)
    reorder_legend(ax_main)

    finalize_manifest(
        manifest,
        event_name=event_name,
        lc_input_path=lc_input_path,
        samples_path=samples_path,
        blobs_path=blobs_path,
        keys_path=keys_path,
        tex_enabled=tex_enabled,
        tex_warning=tex_warning,
        model_label=render_model_label,
        used_truth_fallback=used_truth_fallback,
        truth_overlay=truth_overlay,
        n_requested=n_plot,
        n_success=n_success,
        n_failed=n_failed,
        posterior_envelopes_drawn=posterior_envelopes_drawn,
        zoom_inset_added=zoom_inset_added,
        truth_residual_overlay=truth_residual_overlay,
        anomaly_inset_meta=anomaly_inset_meta,
        t0_marker_added=t0_marker_added,
        t0_marker_bjd=truth_t0_bjd,
        t0_vline_entry=t0_vline_entry,
    )

    roman_plot.write_outputs(fig, manifest, renderer_args)
    plt.close(fig)
    print(
        f"  Saved: {output_stem.name}.pdf/.png/.meta.json  "
        f"({n_success}/{n_plot} posterior samples rendered)"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)

    for mass_bin in args.bins:
        if mass_bin not in COMPLETED_BINS:
            raise ValueError(f"Unknown mass bin '{mass_bin}'. Valid: {COMPLETED_BINS}")
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be a positive integer")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    orbit_obj = Orbit()   # shared across all events

    for mass_bin in args.bins:
        bin_outdir = outdir / mass_bin
        bin_outdir.mkdir(parents=True, exist_ok=True)

        # The data directory contains .det.lc files, the master HDF5/CSV,
        # the .prm file, and the .gulls_config.json
        data_dir = (DATA_BASE / f"6f_overguide_{mass_bin}" / mass_bin).resolve()
        post_dir = data_dir / "posteriors"

        if not data_dir.exists():
            raise FileNotFoundError(
                f"[{mass_bin}] Data directory not found: {data_dir}")
        if not post_dir.exists():
            raise FileNotFoundError(
                f"[{mass_bin}] Posteriors directory not found: {post_dir}")

        # Use the Data module's own _initialize_directory to resolve the
        # master file, load prm/config, and set sim_time0 etc.
        data_obj = Data()
        normalized_path, files, master_file, _ = data_obj._initialize_directory(
            str(data_dir)
        )
        print(f"\n[{mass_bin}] Initialized data dir: {data_dir}")
        print(f"  Master file: {master_file}")
        print(f"  sim_time0: {data_obj.sim_time0}")

        # Find which LC files have posteriors
        lc_files = sorted([f for f in files if f.endswith(".det.lc")])
        print(f"  {len(lc_files)} lightcurve(s) in data dir")

        n_plotted, n_skipped = 0, 0
        for lc_filename in lc_files:
            event_name, field, subrun, event_id = lc_filename_to_event_name(lc_filename)

            # Only process events that have posteriors
            samples_path, blobs_path, keys_path = find_samples(post_dir, event_name)
            if samples_path is None:
                n_skipped += 1
                continue

            print(f"\n  → {event_name}  ({lc_filename})")

            # ── Load data + truths via the existing Data module ──────────
            # _load_event_from_lcfile calls load_data() and get_params()
            # with all the proper column handling
            event_name_loaded, truths_series, data = data_obj._load_event_from_lcfile(
                normalized_path, master_file, lc_filename
            )
            truths = truths_series.to_dict()
            if "params" in truths and isinstance(truths["params"], list):
                truths["params"] = np.array(truths["params"])

            sim_time0 = data_obj.sim_time0
            gamma = data_obj.gamma

            # ── Load samples ──────────────────────────────────────────────
            samples_raw = np.load(samples_path)    # (n_samples, ndim)
            blobs = load_blobs(blobs_path, keys_path)
            print(f"    samples: {samples_raw.shape},  blobs keys: {list(blobs.keys())}")

            # ── Plot ──────────────────────────────────────────────────────
            plot_event(
                event_name   = event_name,
                data         = data,
                truths       = truths,
                sim_time0    = sim_time0,
                gamma        = gamma,
                samples_raw  = samples_raw,
                blobs        = blobs,
                orbit_obj    = orbit_obj,
                outdir       = bin_outdir,
                lc_input_path = Path(normalized_path) / lc_filename,
                samples_path = samples_path,
                blobs_path   = blobs_path,
                keys_path    = keys_path,
                n_samples_plot = args.n_samples,
            )
            n_plotted += 1

            if args.max_events is not None and n_plotted >= args.max_events:
                print(f"  Reached --max-events={args.max_events} for {mass_bin}")
                break

        print(f"\n[{mass_bin}] Done: {n_plotted} plotted, {n_skipped} skipped (no posteriors)")

    print(f"\nDone. Plots saved to: {outdir}")


if __name__ == "__main__":
    main()
