#!/usr/bin/env python3
"""Full-test residual distributions for the O24 slide's saved cold seed 40442.

This is inference only. It reuses the canonical model, noise=-1 test loading,
and residual definition from make_perf_plots_o22.py without importing that
script (which executes its entire evaluation at import time). The prediction
cache permits plot revisions without repeating inference.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/perf_plots_o24_slides_20260909/residual_40442/figures"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "4"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/smartpix-o24-residual-mpl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINT = ROOT / "runs/o24_scratch_bias/s40442/seed_40442/best.weights.hdf5"
TFR = Path("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_"
           "10ps_300k_convolved_to_200ps/shuffled_3d/"
           "TFR_files_all101_noise_contained_discovery")
CACHE = OUT / "predictions_cold_40442.npz"
REFERENCE_PATH = ROOT / "runs/perf_plots_o24_seeds/perf_summary_o22.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())["cold_40442"]
SHA = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
COLOR = "#7c3aed"
ORANGE = "#c2410c"
TARGETS = [("x", 0, "µm", "x [um]"), ("y", 1, "µm", "y [um]"),
           ("alpha", 2, "deg", "alpha[deg]"), ("beta", 3, "deg", "beta[deg]")]


def predict():
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as cached:
            assert str(cached["checkpoint_sha256"]) == SHA, "Cache belongs to another checkpoint"
            assert len(cached["truth"]) == len(cached["predictions"]) == 37919
            print(f"Reusing all-event prediction cache: {CACHE}", flush=True)
            return {key: cached[key] for key in cached.files}
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    sys.path.insert(0, str(ROOT / "two_bit_optimization_helpers"))
    from prepare_tfrecords import load_tfrecords
    from train import create_model
    from conditional_nll import custom_loss_v2

    # Match the existing canonical evaluation, including seed and noise flag.
    _, test = load_tfrecords(str(TFR / "TFR_train"), str(TFR / "TFR_test"),
                            noise=-1, seed=42)
    scale = np.asarray(json.loads((TFR / "TFR_test/metadata.json").read_text())["labels_scale"],
                       dtype=np.float64)
    model = create_model("ViT_MaxDeep_PairLattice", timeslices=101,
                         soft_quantize_layer=True, initial_thresholds=[1., 2., 3.],
                         threshold_offset=0.0, initial_levels=np.array([0., 1., 2., 3.], np.float32))
    model.load_weights(str(CHECKPOINT))
    pair = np.asarray(model.get_layer("simple_router_output").selected_indices())
    thresholds = np.asarray(model.get_layer("soft_quantizer_output").thresholds)
    assert pair.tolist() == [8, 20]
    assert np.allclose(thresholds, REFERENCE["thresholds"], atol=.0051)
    predictions, truth, batch_sizes = [], [], []
    print(f"CPU inference: {len(test)} complete test batches; pair={pair.tolist()}", flush=True)
    for index in range(len(test)):
        inputs, labels = test[index]
        predictions.append(np.asarray(model.predict_on_batch(inputs)))
        truth.append(np.asarray(labels))
        batch_sizes.append(len(labels))
        print(f"Batch {index + 1}/{len(test)} complete; events={sum(batch_sizes):,}", flush=True)
    predictions = np.concatenate(predictions, axis=0)
    truth = np.concatenate(truth, axis=0)
    assert len(truth) == len(predictions) == 37919
    nll = float(custom_loss_v2(tf.constant(truth, tf.float32),
                              tf.constant(predictions, tf.float32))) / len(test)
    values = dict(predictions=predictions, truth=truth, labels_scale=scale,
                  batch_sizes=np.asarray(batch_sizes), plain_nll=np.asarray(nll),
                  checkpoint_sha256=np.asarray(SHA), slices=pair, thresholds_mv=thresholds)
    np.savez_compressed(CACHE, **values)
    print(f"Cached all {len(truth):,} predictions: {CACHE}", flush=True)
    return values


def inv_cot_canonical(c):
    # Identical branch convention to make_perf_plots_o22.py, including float dtype.
    with np.errstate(divide="ignore"):
        angle = np.arctan(1.0 / c)
    return np.where(angle < 0, angle + np.pi, angle)


def residual_arrays(cached):
    result = {}
    p, y, scale = cached["predictions"], cached["truth"], cached["labels_scale"]
    for name, index, unit, reference_key in TARGETS:
        if index < 2:
            predicted, actual = p[:, 2 * index] * scale[index], y[:, index] * scale[index]
        else:
            predicted = inv_cot_canonical(p[:, 2 * index] * scale[index]) * 180 / np.pi
            actual = inv_cot_canonical(y[:, index] * scale[index]) * 180 / np.pi
        residual = actual - predicted
        sigma = float(np.std(residual, ddof=0))
        assert abs(sigma - REFERENCE["resolution_std"][reference_key]) < 1e-4, (name, sigma)
        result[name] = residual.astype(np.float64)
    return result


def describe(r):
    mean, sigma = float(r.mean()), float(r.std(ddof=0))
    q16, q50, q84 = (float(v) for v in np.quantile(r, [.16, .50, .84]))
    variance_sum = float(np.sum((r - mean) ** 2))
    tails = {}
    for threshold in [5, 10, 15, 20]:
        mask = np.abs(r) > threshold
        tails[f"abs_residual_gt_{threshold}"] = dict(
            threshold=float(threshold), count=int(mask.sum()), fraction=float(mask.mean()),
            contribution_to_total_centered_variance=float(np.sum((r[mask] - mean) ** 2) / variance_sum))
    mask = np.abs(r - mean) > 2 * sigma
    tails["distance_from_mean_gt_2_global_sigma"] = dict(
        threshold=float(2 * sigma), count=int(mask.sum()), fraction=float(mask.mean()),
        contribution_to_total_centered_variance=float(np.sum((r[mask] - mean) ** 2) / variance_sum))
    return dict(events=len(r), mean=mean, global_std_ddof0=sigma,
                median=q50, q16=q16, q84=q84, central68_half_width=(q84 - q16) / 2,
                minimum=float(r.min()), maximum=float(r.max()),
                quantiles={str(q): float(v) for q, v in zip(
                    [.001, .01, .025, .05, .95, .975, .99, .999],
                    np.quantile(r, [.001, .01, .025, .05, .95, .975, .99, .999]))}, tails=tails)


def styling():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.labelsize": 15, "axes.titlesize": 17,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#64748b", "axes.labelcolor": "#1e293b",
                         "xtick.color": "#334155", "ytick.color": "#334155",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})


def histogram(ax, residual, stat, label, bin_width, *, log=True, zoom=None):
    lower = np.floor(residual.min() / bin_width) * bin_width
    upper = np.ceil(residual.max() / bin_width) * bin_width
    edges = np.arange(lower, upper + 1.001 * bin_width, bin_width)
    counts, edges = np.histogram(residual, bins=edges)
    assert counts.sum() == len(residual), "Histogram dropped events"
    ax.stairs(counts, edges, color=COLOR, linewidth=1.5)
    ax.axvspan(stat["q16"], stat["q84"], color=COLOR, alpha=.12, label="16th–84th percentiles")
    ax.axvline(stat["mean"], color="#334155", lw=1.1, label="Mean")
    for sign in [-1, 1]:
        ax.axvline(stat["mean"] + sign * stat["global_std_ddof0"],
                   color=ORANGE, ls="--", lw=1.35, label="Mean ± global std" if sign == 1 else None)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(.7, max(counts) * 1.7)
    else:
        ax.set_ylim(0, max(counts) * 1.12)
    ax.set_xlim(*(zoom if zoom else (edges[0], edges[-1])))
    ax.set_xlabel(label)
    ax.set_ylabel(f"Events / {bin_width:g} {'µm' if 'µm' in label else 'degree'}")
    ax.grid(axis="y", alpha=.18)


def save(fig, stem):
    for ext in ["png", "pdf"]:
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        print(f"Wrote {path}", flush=True)
    plt.close(fig)


def figures(arrays, stats):
    styling()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(top=.95, bottom=.13, left=.085, right=.985, hspace=.48, wspace=.23)
    for ax, (name, _, unit, _) in zip(axes.flat, TARGETS):
        stat = stats[name]
        symbol = {"alpha": "α", "beta": "β"}.get(name, name)
        histogram(ax, arrays[name], stat, f"True − predicted {symbol} [{unit}]", 1 if name in ["x", "alpha", "beta"] else .5)
        ax.set_title(f"{symbol}: global std = {stat['global_std_ddof0']:.2f} {unit}", loc="left", weight="bold")
        ax.text(.98, .96, f"Mean {stat['mean']:+.2f} {unit}\nCentral 68% half-width {stat['central68_half_width']:.2f} {unit}",
                transform=ax.transAxes, va="top", ha="right", fontsize=12,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 4})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .012), ncol=3, frameon=False)
    save(fig, "residuals_cold_40442")

    r, st = arrays["beta"], stats["beta"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.3))
    fig.subplots_adjust(top=.90, bottom=.24, left=.065, right=.985, wspace=.21)
    histogram(axes[0], r, st, "True − predicted β [deg]", 1, log=True)
    axes[0].set_title("Full residual range · logarithmic counts", loc="left", fontsize=17)
    zoom = (-15, 15)
    histogram(axes[1], r, st, "True − predicted β [deg]", .25, log=False, zoom=zoom)
    inside = int(((r >= zoom[0]) & (r <= zoom[1])).sum())
    axes[1].set_title("Central view ±15° · linear counts", loc="left", fontsize=17)
    axes[1].text(.98, .95, f"{inside:,} / {len(r):,}\nevents in view", ha="right", va="top",
                 transform=axes[1].transAxes, fontsize=14,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 4})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, frameon=False)
    save(fig, "beta_residual_cold_40442")


def main():
    cached = predict()
    nll = float(cached["plain_nll"])
    assert abs(nll - REFERENCE["plain_val_nll"]) < .10, (nll, REFERENCE["plain_val_nll"])
    arrays = residual_arrays(cached)
    stats = {name: describe(values) for name, values in arrays.items()}
    metrics = dict(arm="cold_40442", events=37919, plain_nll=nll,
                   residual_definition="true - predicted; angular inverse cotangent in [0,180] degrees",
                   std_definition="Population standard deviation over all events, ddof=0",
                   central68_definition="16th to 84th percentiles; half-width=(q84-q16)/2; not Gaussian-fitted",
                   tail_variance_definition="sum((residual-global_mean)^2) for tail events / same sum for all events",
                   residuals=stats)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    provenance = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      script=str(Path(__file__).resolve()), checkpoint=str(CHECKPOINT),
                      checkpoint_sha256=SHA, epoch_one_based=7576, selection="best composite val_loss",
                      source_slide="O24 07/18", dataset=str(TFR / "TFR_test"), noise=-1, seed=42,
                      model="ViT_MaxDeep_PairLattice", cpu_only=True, tensorflow_threads=4,
                      batch_sizes=cached["batch_sizes"].tolist(), events=37919,
                      prediction_cache=str(CACHE), reference_metrics=str(REFERENCE_PATH),
                      reference_key="cold_40442", std_verification_tolerance=1e-4,
                      nll_verification_tolerance=.10, actual_nll_difference=nll-REFERENCE["plain_val_nll"],
                      angular_note="Same inverse-cotangent implementation and float32 event residuals as canonical performance script; statistics accumulated in float64.",
                      plots_use_all_events=True, central_view_note="Zoom changes axis limits only; global statistics and histogram counts use every event.")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    figures(arrays, stats)
    print(json.dumps(metrics, indent=2), flush=True)
    print("RESIDUAL FIGURES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
