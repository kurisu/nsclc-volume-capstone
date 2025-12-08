#!/usr/bin/env python3
"""
Analyze nnU-Net training logs for a single fold (fold_0) and generate:
- Aggregated CSV of epoch-level metrics
- Line plots for Pseudo Dice, Losses, Learning Rate, and Epoch Time

Usage:
  python scripts/analyze_nnunet_training.py \
    --input-dir /Users/kai/workspace/capstone_transfer/capstone_storage/nnunet/nnUNet_results/Dataset501_NSCLC_Lung1/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0 \
    --output-root /Users/kai/workspace/capstone

Notes:
- If matplotlib/seaborn are not available, the script will still write the CSV.
- Epoch metrics are parsed from training_log_*.txt files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class EpochMetrics:
    epoch: int
    learning_rate: Optional[float] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    pseudo_dice: Optional[float] = None
    epoch_time_s: Optional[float] = None
    best_ema_pseudo_dice: Optional[float] = None  # value at the time this epoch was processed


EPOCH_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}.*?: Epoch (\d+)\s*$")
LR_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}.*?: Current learning rate: ([0-9.]+)\s*$")
TRAIN_LOSS_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}.*?: train_loss (-?[0-9.]+)\s*$")
VAL_LOSS_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}.*?: val_loss (-?[0-9.]+)\s*$")
PSEUDO_DICE_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}.*?: Pseudo dice \[np\.float32\(([0-9.]+)\)\]\s*$"
)
EPOCH_TIME_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}.*?: Epoch time: ([0-9.]+) s\s*$")
BEST_EMA_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}.*?: Yayy! New best EMA pseudo Dice: ([0-9.]+)\s*$"
)


def parse_logs(input_dir: Path) -> Tuple[Dict[int, EpochMetrics], Optional[float]]:
    """
    Parse all training_log_*.txt files in input_dir and return:
    - epoch_to_metrics mapping
    - best_ema_overall (float if encountered)
    """
    epoch_to_metrics: Dict[int, EpochMetrics] = {}
    best_ema_overall: Optional[float] = None

    log_files = sorted(input_dir.glob("training_log_*.txt"))
    if not log_files:
        raise FileNotFoundError(f"No training_log_*.txt files found in: {input_dir}")

    current_epoch: Optional[int] = None
    for log_file in log_files:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Epoch
                m = EPOCH_RE.match(line)
                if m:
                    current_epoch = int(m.group(1))
                    epoch_to_metrics.setdefault(current_epoch, EpochMetrics(epoch=current_epoch))
                    continue

                # LR
                m = LR_RE.match(line)
                if m and current_epoch is not None:
                    epoch_to_metrics[current_epoch].learning_rate = float(m.group(1))
                    continue

                # train_loss
                m = TRAIN_LOSS_RE.match(line)
                if m and current_epoch is not None:
                    epoch_to_metrics[current_epoch].train_loss = float(m.group(1))
                    continue

                # val_loss
                m = VAL_LOSS_RE.match(line)
                if m and current_epoch is not None:
                    epoch_to_metrics[current_epoch].val_loss = float(m.group(1))
                    continue

                # pseudo dice
                m = PSEUDO_DICE_RE.match(line)
                if m and current_epoch is not None:
                    epoch_to_metrics[current_epoch].pseudo_dice = float(m.group(1))
                    continue

                # epoch time
                m = EPOCH_TIME_RE.match(line)
                if m and current_epoch is not None:
                    epoch_to_metrics[current_epoch].epoch_time_s = float(m.group(1))
                    continue

                # best EMA pseudo Dice
                m = BEST_EMA_RE.match(line)
                if m:
                    best_ema_overall = float(m.group(1))
                    if current_epoch is not None:
                        epoch_to_metrics[current_epoch].best_ema_pseudo_dice = best_ema_overall

    return epoch_to_metrics, best_ema_overall


def try_import_plotting():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns  # noqa: F401
        return True
    except Exception:
        return False


def write_csv(epoch_to_metrics: Dict[int, EpochMetrics], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    fieldnames = list(asdict(EpochMetrics(epoch=0)).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in sorted(epoch_to_metrics):
            writer.writerow(asdict(epoch_to_metrics[epoch]))


def compute_aggregates(epoch_to_metrics: Dict[int, EpochMetrics]) -> Dict[str, float]:
    epochs_sorted = sorted(epoch_to_metrics)
    if not epochs_sorted:
        return {}
    last_n = 20
    last_epochs = epochs_sorted[-last_n:] if len(epochs_sorted) >= last_n else epochs_sorted

    def safe_values(selector):
        vals = [selector(epoch_to_metrics[e]) for e in epochs_sorted]
        return [v for v in vals if v is not None]

    def safe_values_last(selector):
        vals = [selector(epoch_to_metrics[e]) for e in last_epochs]
        return [v for v in vals if v is not None]

    import statistics as stats

    pseudo_all = safe_values(lambda m: m.pseudo_dice)
    val_loss_all = safe_values(lambda m: m.val_loss)
    train_loss_all = safe_values(lambda m: m.train_loss)
    lr_all = safe_values(lambda m: m.learning_rate)
    t_all = safe_values(lambda m: m.epoch_time_s)

    best_pseudo = max(pseudo_all) if pseudo_all else float("nan")
    best_epoch = (
        max((e for e in epochs_sorted if epoch_to_metrics[e].pseudo_dice == best_pseudo), default=None)
        if pseudo_all
        else None
    )
    best_vloss = min(val_loss_all) if val_loss_all else float("nan")
    best_vloss_epoch = (
        min((e for e in epochs_sorted if epoch_to_metrics[e].val_loss == best_vloss), default=None)
        if val_loss_all
        else None
    )

    pseudo_last = safe_values_last(lambda m: m.pseudo_dice)
    val_loss_last = safe_values_last(lambda m: m.val_loss)
    train_loss_last = safe_values_last(lambda m: m.train_loss)
    t_last = safe_values_last(lambda m: m.epoch_time_s)

    agg = {
        "epoch_min": float(min(epochs_sorted)),
        "epoch_max": float(max(epochs_sorted)),
        "num_epochs_parsed": float(len(epochs_sorted)),
        "best_pseudo_dice": float(best_pseudo),
        "best_pseudo_dice_epoch": float(best_epoch) if best_epoch is not None else float("nan"),
        "best_val_loss": float(best_vloss),
        "best_val_loss_epoch": float(best_vloss_epoch) if best_vloss_epoch is not None else float("nan"),
        "pseudo_dice_mean_last20": float(stats.mean(pseudo_last)) if pseudo_last else float("nan"),
        "pseudo_dice_median_last20": float(stats.median(pseudo_last)) if pseudo_last else float("nan"),
        "val_loss_mean_last20": float(stats.mean(val_loss_last)) if val_loss_last else float("nan"),
        "train_loss_mean_last20": float(stats.mean(train_loss_last)) if train_loss_last else float("nan"),
        "epoch_time_mean": float(stats.mean(t_all)) if t_all else float("nan"),
        "epoch_time_mean_last20": float(stats.mean(t_last)) if t_last else float("nan"),
        "learning_rate_min": float(min(lr_all)) if lr_all else float("nan"),
        "learning_rate_max": float(max(lr_all)) if lr_all else float("nan"),
    }
    return agg


def maybe_plot(epoch_to_metrics: Dict[int, EpochMetrics], fig_dir: Path) -> List[Path]:
    if not try_import_plotting():
        return []

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("whitegrid")
    fig_dir.mkdir(parents=True, exist_ok=True)

    epochs = sorted(epoch_to_metrics)
    pseudo = [epoch_to_metrics[e].pseudo_dice for e in epochs]
    vloss = [epoch_to_metrics[e].val_loss for e in epochs]
    tloss = [epoch_to_metrics[e].train_loss for e in epochs]
    lrs = [epoch_to_metrics[e].learning_rate for e in epochs]
    etimes = [epoch_to_metrics[e].epoch_time_s for e in epochs]

    outputs: List[Path] = []

    # Pseudo Dice
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=pseudo, marker="o", linewidth=1.5)
    plt.title("Validation Pseudo Dice vs Epoch (fold_0)")
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo Dice")
    plt.tight_layout()
    out = fig_dir / "nnunet_fold0_pseudodice.png"
    plt.savefig(out, dpi=144)
    plt.close()
    outputs.append(out)

    # Losses
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=tloss, label="train_loss", linewidth=1.5)
    sns.lineplot(x=epochs, y=vloss, label="val_loss", linewidth=1.5)
    plt.title("Losses vs Epoch (fold_0)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    out = fig_dir / "nnunet_fold0_losses.png"
    plt.savefig(out, dpi=144)
    plt.close()
    outputs.append(out)

    # Learning Rate
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=lrs, linewidth=1.5)
    plt.title("Learning Rate vs Epoch (fold_0)")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.tight_layout()
    out = fig_dir / "nnunet_fold0_lr.png"
    plt.savefig(out, dpi=144)
    plt.close()
    outputs.append(out)

    # Epoch time
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=etimes, linewidth=1.5)
    plt.title("Epoch Time vs Epoch (fold_0)")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    out = fig_dir / "nnunet_fold0_epoch_time.png"
    plt.savefig(out, dpi=144)
    plt.close()
    outputs.append(out)

    return outputs


def save_aggregates_json(aggregates: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)


def read_debug_best_ema(input_dir: Path) -> Optional[float]:
    dbg_path = input_dir / "debug.json"
    if not dbg_path.exists():
        return None
    try:
        dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
        val = dbg.get("_best_ema", None)
        return float(val) if val is not None else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze nnU-Net fold_0 logs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to fold_0 directory containing training_log_*.txt and progress.png",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Project root where data/processed and reports/figures exist or will be created",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_root: Path = args.output_root

    epoch_to_metrics, best_ema_from_logs = parse_logs(input_dir)
    aggregates = compute_aggregates(epoch_to_metrics)

    # Also capture _best_ema from debug.json if present
    best_ema_from_debug = read_debug_best_ema(input_dir)
    if best_ema_from_logs is not None:
        aggregates["best_ema_pseudo_dice_from_logs"] = float(best_ema_from_logs)
    if best_ema_from_debug is not None:
        aggregates["best_ema_pseudo_dice_from_debug"] = float(best_ema_from_debug)

    # Write CSV
    csv_path = output_root / "data" / "processed" / "nnunet_fold0_metrics.csv"
    write_csv(epoch_to_metrics, csv_path)

    # Plot (if possible)
    fig_dir = output_root / "reports" / "figures"
    plot_paths = maybe_plot(epoch_to_metrics, fig_dir)

    # Save aggregates JSON for convenience
    agg_path = output_root / "data" / "processed" / "nnunet_fold0_aggregates.json"
    save_aggregates_json(aggregates, agg_path)

    print("Aggregates:")
    for k, v in aggregates.items():
        print(f"  {k}: {v}")
    print(f"\nCSV written: {csv_path}")
    if plot_paths:
        print("Plots:")
        for p in plot_paths:
            print(f"  {p}")
    else:
        print("Plots skipped (matplotlib/seaborn not available).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


