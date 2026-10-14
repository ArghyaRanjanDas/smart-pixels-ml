#!/usr/bin/env python3
"""Linear, filled residual histograms adapted from from_weights.ipynb.

Notebook source cells 18/22/26/30 (zero-based) use plt.hist, 50 equally
spaced edges (49 bins), default event counts and linear y axes. This adaptation
uses the existing complete O24 inference cache, correct current label scales,
and angle residuals in degrees to match the slide's resolution table. Display
windows are explicit; bin edges extend over all events before axes are zoomed.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/smartpix-o24-linear-residual-mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/perf_plots_o24_slides_20260909/residual_linear_40442/figures"
CACHE = ROOT / "runs/perf_plots_o24_slides_20260909/residual_40442/figures/predictions_cold_40442.npz"
CHECKPOINT = ROOT / "runs/o24_scratch_bias/s40442/seed_40442/best.weights.hdf5"
NOTEBOOK = ROOT / "legacy/notebooks/physics/from_weights.ipynb"
REFERENCE = ROOT / "runs/perf_plots_o24_seeds/perf_summary_o22.json"
WINDOWS = {"x": (-40., 40.), "y": (-12., 12.), "alpha": (-15., 15.), "beta": (-15., 15.)}
UNITS = {"x": "µm", "y": "µm", "alpha": "deg", "beta": "deg"}
SYMBOLS = {"x": "x", "y": "y", "alpha": "α", "beta": "β"}
KEYS = {"x": "x [um]", "y": "y [um]", "alpha": "alpha[deg]", "beta": "beta[deg]"}
PURPLE = "#7c3aed"


def inverse_cot(c):
    with np.errstate(divide="ignore"):
        a = np.arctan(1.0 / c)
    return np.where(a < 0, a + np.pi, a)


def load_residuals():
    expected_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    with np.load(CACHE, allow_pickle=False) as cache:
        assert str(cache["checkpoint_sha256"]) == expected_sha
        p, y, scale = cache["predictions"], cache["truth"], cache["labels_scale"]
        assert p.shape == (37919, 14) and y.shape == (37919, 4)
        metadata = {"checkpoint_sha256": expected_sha, "labels_scale": scale.tolist(),
                    "events": len(y), "plain_nll": float(cache["plain_nll"]),
                    "batch_sizes": cache["batch_sizes"].tolist()}
        residuals = {}
        ref = json.loads(REFERENCE.read_text())["cold_40442"]
        for index, name in enumerate(WINDOWS):
            pred, true = p[:, 2 * index] * scale[index], y[:, index] * scale[index]
            if index >= 2:
                pred, true = inverse_cot(pred) * 180 / np.pi, inverse_cot(true) * 180 / np.pi
            r = (true - pred).astype(np.float64)
            assert np.isfinite(r).all() and len(r) == 37919
            assert abs(r.std(ddof=0) - ref["resolution_std"][KEYS[name]]) < 1e-4
            residuals[name] = r
    return residuals, metadata


def full_edges(residuals, central_window):
    """Extend the notebook's 49 equal central bins over every observed value."""
    left, right = central_window
    width = (right - left) / 49
    first = min(0, int(np.floor((residuals.min() - left) / width)))
    last = max(49, int(np.ceil((residuals.max() - left) / width)))
    edges = left + np.arange(first, last + 1) * width
    assert edges[0] <= residuals.min() and edges[-1] >= residuals.max()
    return edges


def describe(residuals, window):
    lo, hi = window
    q16, q84 = np.quantile(residuals, [.16, .84])
    mean = float(residuals.mean())
    visible = (residuals >= lo) & (residuals <= hi)
    variance_sum = np.sum((residuals - mean) ** 2)
    tails = {}
    for threshold in [10, 15]:
        mask = np.abs(residuals) > threshold
        tails[f"abs_residual_gt_{threshold}"] = dict(
            count=int(mask.sum()), fraction=float(mask.mean()),
            contribution_to_total_centered_variance=float(np.sum((residuals[mask] - mean) ** 2) / variance_sum))
    return {"events": len(residuals), "mean": mean, "global_std_ddof0": float(residuals.std(ddof=0)),
            "q16": float(q16), "q84": float(q84), "central68_half_width": float((q84-q16)/2),
            "minimum": float(residuals.min()), "maximum": float(residuals.max()),
            "display_window": [lo, hi], "events_in_view": int(visible.sum()),
            "events_outside_view": int((~visible).sum()), "fraction_outside_view": float((~visible).mean()),
            "central_bins": 49, "bin_width": (hi-lo)/49, "tails": tails}


def draw_hist(ax, residuals, name, stats, edges, *, zoom):
    counts, actual_edges, _ = ax.hist(residuals, bins=edges, density=False,
                                      color=PURPLE, alpha=.82, edgecolor="white", linewidth=.2)
    assert counts.sum() == len(residuals) == 37919
    # Set explicitly: source notebook comments out the logarithmic y scaling.
    ax.set_yscale("linear")
    ax.set_ylim(0, counts.max() * 1.10)
    ax.set_xlim(*(WINDOWS[name] if zoom else (actual_edges[0], actual_edges[-1])))
    ax.axvline(stats["mean"], color="#334155", linewidth=1.0, label="Mean")
    for sign in [-1, 1]:
        ax.axvline(stats["mean"] + sign * stats["global_std_ddof0"], color="#c2410c",
                   linestyle="--", linewidth=1.1,
                   label="Mean ± global std" if sign == 1 else None)
    ax.set_xlabel(f"True − predicted {SYMBOLS[name]} [{UNITS[name]}]")
    ax.set_ylabel(f"Events / {stats['bin_width']:.3g} {'µm' if name in ['x', 'y'] else 'degree'}")
    ax.grid(axis="y", alpha=.18)
    return int(counts.sum())


def save(fig, stem):
    for extension in ["png", "pdf"]:
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
        print(path, flush=True)
    plt.close(fig)


def make_figures(residuals, stats):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.labelsize": 15, "axes.titlesize": 16,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#64748b", "axes.labelcolor": "#1e293b",
                         "xtick.color": "#334155", "ytick.color": "#334155", "pdf.fonttype": 42})
    edges = {name: full_edges(r, WINDOWS[name]) for name, r in residuals.items()}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(top=.93, bottom=.13, left=.09, right=.985, hspace=.60, wspace=.25)
    for ax, name in zip(axes.flat, WINDOWS):
        st, r = stats[name], residuals[name]
        total = draw_hist(ax, r, name, st, edges[name], zoom=True)
        st["histogram_events_all_bins"] = total
        st["full_histogram_edge_range"] = [float(edges[name][0]), float(edges[name][-1])]
        st["full_histogram_bins"] = len(edges[name])-1
        ax.set_title(f"{SYMBOLS[name]} residual · linear counts\n{st['events_in_view']:,} in view; {st['events_outside_view']:,} outside",
                     loc="left", fontsize=15)
        ax.text(.98, .96, f"Mean {st['mean']:+.2f} {UNITS[name]}\nGlobal std {st['global_std_ddof0']:.2f} {UNITS[name]}\n68% half-width {st['central68_half_width']:.2f} {UNITS[name]}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), ncol=2, frameon=False)
    save(fig, "residuals_linear_cold_40442")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.3))
    fig.subplots_adjust(top=.86, bottom=.24, left=.07, right=.985, wspace=.22)
    name = "beta"
    st, r = stats[name], residuals[name]
    for ax, zoom in zip(axes, [False, True]):
        draw_hist(ax, r, name, st, edges[name], zoom=zoom)
    axes[0].set_title("Full observed range · linear counts\nAll 37,919 events", loc="left", fontsize=16)
    axes[1].set_title(f"Central view ±15° · linear counts\n{st['events_in_view']:,} in view; {st['events_outside_view']:,} outside", loc="left", fontsize=16)
    axes[1].text(.98, .95, f"Mean {st['mean']:+.3f}°\nGlobal std {st['global_std_ddof0']:.3f}°\n68% half-width {st['central68_half_width']:.3f}°",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=13,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=2, frameon=False)
    save(fig, "beta_residual_linear_cold_40442")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    residuals, cache_metadata = load_residuals()
    stats = {name: describe(r, WINDOWS[name]) for name, r in residuals.items()}
    make_figures(residuals, stats)
    metrics = {"arm": "cold_40442", **cache_metadata, "residuals": stats,
               "residual_definition": "true - predicted; x/y in micrometres, alpha/beta in degrees",
               "global_std_definition": "Population std over all 37,919 event residuals, ddof=0; no fit and no range cut",
               "central68_definition": "16th–84th percentiles of all events; half-width=(q84-q16)/2",
               "histogram_definition": "Linear filled event-count histogram, density=False; 49 equal central bins extended over the entire observed range. Axes only are zoomed.",
               "full_beta_uses_same_bin_edges_as_central": True}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    provenance = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "script": str(Path(__file__).resolve()), "checkpoint": str(CHECKPOINT),
                  "checkpoint_sha256": cache_metadata["checkpoint_sha256"],
                  "checkpoint_epoch_one_based": 7576, "selection": "saved best composite val_loss",
                  "prediction_cache": str(CACHE), "inference_rerun": False, "training_run": False,
                  "events": 37919, "histograms_and_statistics_include_all_events": True,
                  "notebook_source": str(NOTEBOOK),
                  "notebook_cells_zero_based": {"residual_definitions": 10, "mean_and_std": [12,13,14,15],
                                                "linear_histograms": [18,22,26,30]},
                  "notebook_histogram_conventions": {"function": "plt.hist", "edges": 50, "bins": 49,
                                                      "density": False, "y_scale": "linear", "histtype": "bar", "fit": None},
                  "adaptations": [
                      "Uses current metadata label scales instead of the notebook's legacy hardcoded 75,18.75,8,0.5.",
                      "Angle residuals are converted to degrees to match current slide tables; notebook histogram cells use cotangent residuals.",
                      "Central display windows are x ±40 µm, y ±12 µm, alpha/beta ±15 degrees; these are visualization limits, not event selections.",
                      "Extends central 49-bin spacing to full observed range so histogram counts include every event, unlike fixed-range notebook bins.",
                      "Adds global-statistic references and visible/outside event counts; does not fit a Gaussian."],
                  "labels_scale": cache_metadata["labels_scale"], "comparison_reference": str(REFERENCE),
                  "residual_std_verification_tolerance": 1e-4}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print("LINEAR RESIDUAL FIGURES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
