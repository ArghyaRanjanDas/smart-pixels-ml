"""Reproduce the O24 slide history figures from every recorded epoch.

Run with /work/users/das214/envs/smartpix-2bit/bin/python from this repository.
No training, inference, or event selection is performed. Only unweighted
``plain_nll`` and ``val_plain_nll`` are plotted; composite losses are audited
but never put on the same likelihood axis. Epoch numbers are one-based.

Malformed CSV rows and nonfinite values are reported. The only numerical
rejection is the handoff's absolute bound, abs(value) >= 5e6. No local-median
outlier cut is made. Faint curves contain every retained point; opaque curves
are a labelled 101-epoch centred median for legibility. Companion zoom panels
clip the view, not the source data. A history minimum is not evidence that the
corresponding checkpoint survived a resumed run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BOUND = 5e6
MEDIAN_WINDOW = 101
COLORS = ["#1278A6", "#DA6942", "#42946C", "#8A67AD"]
TRAIN_COLOR, VAL_COLOR = "#527EAD", "#D66039"
CAMPAIGNS = {
    "cold": ("o24_scratch_bias", [40342, 40442, 40542, 40642]),
    "warm": ("o24b_from_o22", [41042, 41142, 41242]),
    "recovery": ("o24b_recover", [41142]),
}


def read_history(path: Path) -> dict:
    epoch, values = [], {key: [] for key in ("plain_nll", "val_plain_nll", "val_loss")}
    audit = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "malformed_rows": [], "rejected_values": []}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                audit["malformed_rows"].append(line)
                continue
            try:
                ep = float(row["epoch"])
                if not np.isfinite(ep) or ep < 0 or ep != int(ep):
                    raise ValueError("invalid epoch")
            except (ValueError, KeyError):
                audit["malformed_rows"].append(line)
                continue
            epoch.append(int(ep) + 1)
            for key, series in values.items():
                try:
                    value = float(row[key])
                except (ValueError, KeyError):
                    value = np.nan
                if not np.isfinite(value) or abs(value) >= BOUND:
                    audit["rejected_values"].append({"line": line, "epoch": int(ep) + 1,
                                                       "column": key, "raw": row.get(key)})
                    value = np.nan
                series.append(value)
    epoch = np.asarray(epoch)
    if len(epoch) == 0 or np.any(np.diff(epoch) <= 0):
        raise ValueError(f"Missing or non-increasing epoch sequence: {path}")
    values = {key: np.asarray(value) for key, value in values.items()}
    audit.update({"rows": len(epoch), "first_epoch": int(epoch[0]),
                  "last_epoch": int(epoch[-1]),
                  "missing_epochs": sorted(set(range(1, int(epoch[-1]) + 1)) - set(epoch))})
    for key, series in values.items():
        index = int(np.nanargmin(series))
        audit[key] = {"min": float(series[index]), "min_epoch": int(epoch[index]),
                      "final": float(series[-1]), "valid_rows": int(np.isfinite(series).sum())}
    return {"epoch": epoch, **values, "audit": audit}


def median(values):
    if len(values) < MEDIAN_WINDOW:
        return values
    width = MEDIAN_WINDOW // 2
    padded = np.pad(values, width, mode="edge")
    return np.nanmedian(np.lib.stride_tricks.sliding_window_view(padded, MEDIAN_WINDOW), axis=1)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.titlesize": 17, "axes.labelsize": 15,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#B3BFCA",
                         "text.color": "#172C3F", "axes.labelcolor": "#172C3F",
                         "xtick.color": "#405568", "ytick.color": "#405568",
                         "savefig.facecolor": "white", "pdf.fonttype": 42,
                         "path.simplify": False})


def axis_style(ax, *, max_epoch=10000, label="Epoch (one-based)"):
    ax.grid(alpha=.19, linewidth=.7)
    ax.set_xlabel(label)
    ax.set_ylabel("Plain NLL / 1,000  (lower is better)")
    ax.set_xlim(1, max_epoch)
    if max_epoch > 20:
        ax.xaxis.set_major_locator(MultipleLocator(2500))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))


def baselines(ax, refs):
    for name, color, ls in (("parent", "#3A4050", "--"), ("o22_ep100", "#8E8E8E", ":")):
        ax.axhline(refs[name]["nll_val"] / 1000, color=color, ls=ls, lw=1.3, zorder=2)


def header(fig, title, subtitle):
    fig.text(.065, .965, title, fontsize=23, weight="bold", va="top")
    fig.text(.065, .91, subtitle, fontsize=13.5, color="#516579", va="top")


def footer(fig, first, second):
    fig.text(.065, .064, first, fontsize=10.5, color="#516579")
    fig.text(.065, .032, second, fontsize=10.5, color="#516579")


def save(fig, out, name):
    for suffix in ("png", "pdf"):
        path = out / f"{name}.{suffix}"
        fig.savefig(path, dpi=165)
        print(path, flush=True)
    plt.close(fig)


def overview(histories, campaign, refs, out):
    seeds = CAMPAIGNS[campaign][1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.285, top=.815, wspace=.21)
    warm = campaign == "warm"
    title = "O24b warm: the best logged likelihood came early" if warm else "O24 cold: four complete 10,000-epoch runs"
    subtitle = ("O22 warm start; inherited router and thresholds free; no smoothing restart."
                if warm else "Bias-aware objective from epoch 1; free pair router and thresholds; smoothing ends at epoch 3,000.")
    header(fig, title, subtitle)
    handles = []
    for seed, color in zip(seeds, COLORS):
        h = histories[f"{campaign}_{seed}"]
        x, raw = h["epoch"], h["val_plain_nll"] / 1000
        trend = median(raw)
        best = int(np.nanargmin(raw))
        for ax in axes:
            ax.plot(x, raw, color=color, lw=.55, alpha=.18, rasterized=True)
            ax.plot(x, trend, color=color, lw=1.9)
            ax.scatter(x[best], raw[best], marker="*", color=color, s=120,
                       edgecolor="white", linewidth=.5, zorder=6)
        handles.append(Line2D([], [], color=color, lw=2,
                             label=f"{seed}: min {raw[best]*1000:,.0f} at epoch {x[best]:,}"))
    for ax in axes:
        baselines(ax, refs)
        axis_style(ax)
    axes[0].set_title("Complete recorded range", loc="left", pad=14)
    axes[1].set_title("Likelihood-region zoom · same 10,000 epochs", loc="left", pad=14)
    axes[1].set_ylim((-41.0, -35.5) if warm else (-41.0, -31.0))
    axes[1].text(.98, .05, f"Baselines — dotted O22: {refs['o22_ep100']['nll_val']:,.0f}\n"
                 f"dashed parent: {refs['parent']['nll_val']:,.0f}",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=11.5,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.94))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.058, .12),
               ncol=2, frameon=False, handlelength=2.7, columnspacing=3)
    footer(fig, "Faint: every raw validation point. Solid: centred 101-epoch median. Stars: logged minima, not proof of a saved checkpoint.",
           "NLL is the mean of batch-summed unweighted loss v2. Horizontal lines: separately evaluated parent / O22 checkpoints. No rejected rows.")
    save(fig, out, f"history_o24_{campaign}")


def early(histories, refs, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.32, top=.815, wspace=.21)
    header(fig, "O24b: the first 12 epochs explain the recovery run",
           "Left: the starts of the original 10,000-epoch runs. Right: a separate 12-epoch run, with plain-NLL checkpoint selection.")
    left, right = axes
    for seed, color in zip(CAMPAIGNS["warm"][1], COLORS):
        h = histories[f"warm_{seed}"]
        x, y = h["epoch"][:12], h["val_plain_nll"][:12] / 1000
        best = int(np.argmin(y))
        left.plot(x, y, "o-", color=color, ms=5, lw=1.6,
                  label=f"{seed}: {y[best]*1000:,.0f} at epoch {x[best]}")
        left.scatter(x[best], y[best], marker="*", s=175, color=color,
                     edgecolor="white", linewidth=.6, zorder=5)
    h = histories["recovery_41142"]
    x, y = h["epoch"], h["val_plain_nll"] / 1000
    best = int(np.argmin(y))
    right.plot(x, y, "o-", color="#1278A6", lw=2, ms=6, label="Recovery 41142 · raw validation")
    right.scatter(x[best], y[best], s=200, marker="*", color="#42946C",
                  edgecolor="white", linewidth=.6, zorder=5)
    saved = refs["recovered"]["nll_val"]
    right.annotate(f"Saved at epoch {x[best]}\nFull-test re-evaluation: {saved:,.0f}",
                   xy=(x[best], y[best]), xytext=(4.5, -40.65), fontsize=12,
                   arrowprops=dict(arrowstyle="->", color="#294F68"),
                   bbox=dict(boxstyle="round,pad=.4", facecolor="#EEF6F4", edgecolor="none"))
    for ax in axes:
        baselines(ax, refs)
        axis_style(ax, max_epoch=12, label="Epoch after the O22 warm start (one-based)")
        ax.set_xticks(range(1, 13))
    left.set_ylim(-40.85, -33.6)
    right.set_ylim(-40.95, -37.0)
    left.set_title("Original 10k runs · historical minima", loc="left", pad=14)
    right.set_title("Recovery · saved best.weights", loc="left", pad=14)
    left.legend(loc="upper center", bbox_to_anchor=(.5, -.19), frameon=False, fontsize=11, ncol=1)
    right.legend(loc="upper center", bbox_to_anchor=(.5, -.19), frameon=False, fontsize=11)
    footer(fig, "Every point in epochs 1–12 is shown without smoothing. The full original histories are in the companion warm-training figure.",
           "Recovery is one seed and a fresh stochastic replay. Dashed / dotted: parent / O22 checkpoints, evaluated on all 37,919 test events.")
    save(fig, out, "history_o24_warm_early")


def seed_history(h, campaign, seed, refs, out):
    recovery = campaign == "recovery"
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.27, top=.81, wspace=.21)
    total = int(h["epoch"][-1])
    best = h["audit"]["val_plain_nll"]
    label = {"cold": "O24 cold", "warm": "O24b warm", "recovery": "O24b recovery"}[campaign]
    header(fig, f"{label} · seed {seed} · {total:,} completed epochs",
           f"Logged validation minimum: {best['min']:,.0f} at epoch {best['min_epoch']:,}. "
           "Training and validation use the same unweighted loss-v2 definition.")
    for ax in axes:
        for key, color, name in (("plain_nll", TRAIN_COLOR, "Training · sampled pairs"),
                                 ("val_plain_nll", VAL_COLOR, "Validation · argmax pair")):
            raw = h[key] / 1000
            if recovery:
                ax.plot(h["epoch"], raw, "o-", ms=5, color=color, lw=1.7, label=name)
            else:
                ax.plot(h["epoch"], raw, color=color, lw=.55, alpha=.18, rasterized=True)
                ax.plot(h["epoch"], median(raw), color=color, lw=1.9, label=name)
        axis_style(ax, max_epoch=total)
        baselines(ax, refs)
    axes[0].set_title("Complete recorded range", loc="left", pad=14)
    axes[1].set_title("Likelihood-region zoom · same epochs", loc="left", pad=14)
    axes[1].set_ylim(-46, -32 if campaign == "cold" else -35)
    axes[1].scatter(best["min_epoch"], best["min"] / 1000, marker="*", color=VAL_COLOR,
                    s=130, edgecolor="white", linewidth=.5, zorder=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.058, .112), ncol=2, frameon=False)
    footer(fig, ("All 12 raw points shown; no smoothing." if recovery else
                 "Faint: every raw point. Solid: centred 101-epoch median. All 10,000 epochs plotted; zoom only changes the displayed y range."),
           "Train: sampled pairs + dropout; validation: argmax pair. Dashed / dotted: parent / O22 checkpoints. Stars: logged minima; saved states may differ.")
    save(fig, out, f"history_o24_{campaign}_{seed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/perf_plots_o24_slides_20260909")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    style()
    source = ROOT / "runs/perf_plots_o24_final/bias_comparison.json"
    refs = json.loads(source.read_text())
    histories = {}
    for campaign, (folder, seeds) in CAMPAIGNS.items():
        for seed in seeds:
            path = ROOT / f"runs/{folder}/s{seed}/seed_{seed}/history.csv"
            h = read_history(path)
            expected = 12 if campaign == "recovery" else 10000
            if h["audit"]["rows"] != expected or h["audit"]["missing_epochs"]:
                raise ValueError(f"Incomplete history: {path}")
            if h["audit"]["malformed_rows"] or h["audit"]["rejected_values"]:
                # Figure captions are valid for this frozen campaign; demand a
                # deliberate caption update if later source revisions are torn.
                raise ValueError(f"Anomalous source history; inspect before plotting: {h['audit']}")
            histories[f"{campaign}_{seed}"] = h
    audit = {"epoch_convention": "one-based (CSV epoch + 1)",
             "numerical_rejection_rule": "nonfinite or abs(value) >= 5e6; no local outlier rejection",
             "median_window_epochs": MEDIAN_WINDOW,
             "reference_checkpoint_source": str(source),
             "reference_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "checkpoint_caveat": "History minimum is not assumed to survive in best.weights; result.json best_epoch is a historical composite-loss minimum.",
             "histories": {key: h["audit"] for key, h in histories.items()},
             "reference_checkpoint_nll": {key: val["nll_val"] for key, val in refs.items()}}
    audit_path = out / "history_audit_o24.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(audit_path, flush=True)
    for campaign in ("cold", "warm"):
        overview(histories, campaign, refs, out)
    early(histories, refs, out)
    for campaign, (_, seeds) in CAMPAIGNS.items():
        for seed in seeds:
            seed_history(histories[f"{campaign}_{seed}"], campaign, seed, refs, out)


if __name__ == "__main__":
    main()
