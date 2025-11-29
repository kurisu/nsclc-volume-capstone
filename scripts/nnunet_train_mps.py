from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Union


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPS-aware wrapper for nnUNetv2 training")
    p.add_argument("dataset_name_or_id", type=str, help="Dataset ID (e.g., 501) or DatasetXXX_Name")
    p.add_argument("configuration", type=str, help="Configuration (e.g., 3d_fullres)")
    p.add_argument("fold", type=str, help="Fold (0..4 or all)")
    p.add_argument("-tr", "--trainer", type=str, default="nnUNetTrainer", help="Trainer class name")
    p.add_argument("-p", "--plans", type=str, default="nnUNetPlans", help="Plans identifier")
    p.add_argument("--pretrained_weights", type=str, default=None, help="Path to pretrained weights")
    p.add_argument("--continue_training", action="store_true", help="Continue training from latest checkpoint")
    p.add_argument("--only_run_validation", action="store_true", help="Skip training, run validation only")
    p.add_argument("--disable_checkpointing", action="store_true", help="Disable checkpointing")
    p.add_argument("--val_with_best", action="store_true", help="Validate with best checkpoint")
    p.add_argument("--export_validation_probabilities", action="store_true", help="Export validation probabilities")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs (default nnU-Net=1000)")
    p.add_argument("--early_stopping", action="store_true", help="Enable simple early stopping controller")
    p.add_argument("--patience_chunks", type=int, default=3, help="Number of chunks without improvement before stop")
    p.add_argument("--chunk_epochs", type=int, default=3, help="Epochs to run per chunk when early stopping is enabled")
    return p.parse_args()


def main() -> None:
    # Force MPS if available; allow CPU fallback for unsupported kernels
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Ensure CUDA path is not selected accidentally
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import torch  # noqa: WPS433 (import inside function is intentional)
    except Exception as e:  # pragma: no cover - helper script
        print("ERROR: PyTorch is not installed. Please install torch with MPS support.", file=sys.stderr)
        raise e

    device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
    if device.type == "mps":
        print("Using device: mps")
    else:
        print("Using device: cpu (MPS unavailable)")

    # Import nnU-Net utilities so we can construct the trainer and override epochs
    try:
        from nnunetv2.run.run_training import (  # noqa: WPS433
            get_trainer_from_args,
            maybe_load_checkpoint,
        )
        from batchgenerators.utilities.file_and_folder_operations import join  # noqa: WPS433
        from torch.backends import cudnn  # noqa: WPS433
    except Exception as e:  # pragma: no cover - helper script
        print(
            "ERROR: nnU-Net v2 is not installed. Install with `pip install nnunetv2`.",
            file=sys.stderr,
        )
        raise e

    args = parse_args()
    # Ensure dataset is passed as string (nnU-Net checks .startswith on it)
    dataset_name_or_id: str
    fold: Union[int, str]
    dataset_name_or_id = str(args.dataset_name_or_id)
    try:
        fold = int(args.fold)
    except ValueError:
        fold = args.fold

    # Build trainer
    nnunet_trainer = get_trainer_from_args(
        dataset_name_or_id=dataset_name_or_id,
        configuration=args.configuration,
        fold=fold if fold != "all" else 0,  # get_trainer_from_args expects int; fold handling happens later
        trainer_name=args.trainer,
        plans_identifier=args.plans,
        device=device,
    )
    # If user requested fold='all', propagate to trainer
    if isinstance(fold, str) and fold == "all":
        nnunet_trainer.fold = "all"
        nnunet_trainer.output_folder = nnunet_trainer.output_folder_base + '_fold_all'
        os.makedirs(nnunet_trainer.output_folder, exist_ok=True)

    # Override epochs if requested
    if args.epochs is not None and args.epochs > 0:
        nnunet_trainer.num_epochs = int(args.epochs)
        print(f"Overriding num_epochs -> {nnunet_trainer.num_epochs}")

    # Disable checkpointing if requested
    if args.disable_checkpointing:
        nnunet_trainer.disable_checkpointing = True

    # Load checkpoint logic (continue / validation-only / pretrained weights)
    if args.continue_training and args.only_run_validation:
        raise RuntimeError("Cannot set both --continue_training and --only_run_validation.")
    maybe_load_checkpoint(
        nnunet_trainer,
        continue_training=args.continue_training,
        validation_only=args.only_run_validation,
        pretrained_weights_file=args.pretrained_weights,
    )

    # CUDNN knobs (safe on non-CUDA)
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    # Helper to get checkpoint_best mtime
    def _best_mtime() -> float:
        ckpt_best = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        try:
            return os.path.getmtime(ckpt_best)
        except FileNotFoundError:
            return 0.0

    # Train with optional early stopping controller
    if not args.only_run_validation:
        if args.early_stopping:
            # Determine target max epochs
            target_max_epochs = int(args.epochs) if args.epochs and args.epochs > 0 else int(nnunet_trainer.num_epochs)
            patience_chunks = max(1, int(args.patience_chunks))
            chunk_epochs = max(1, int(args.chunk_epochs))
            no_improve = 0
            last_best_mtime = _best_mtime()

            while nnunet_trainer.current_epoch < target_max_epochs:
                next_stop = min(target_max_epochs, nnunet_trainer.current_epoch + chunk_epochs)
                nnunet_trainer.num_epochs = next_stop
                print(f"Training chunk: epochs {nnunet_trainer.current_epoch} -> {next_stop}")
                nnunet_trainer.run_training()
                new_best_mtime = _best_mtime()
                if new_best_mtime > last_best_mtime:
                    no_improve = 0
                    last_best_mtime = new_best_mtime
                    print("EarlyStopping: improvement detected (checkpoint_best updated). Reset patience.")
                else:
                    no_improve += 1
                    print(f"EarlyStopping: no improvement this chunk ({no_improve}/{patience_chunks}).")
                    if no_improve >= patience_chunks:
                        print("EarlyStopping: patience exhausted. Stopping training.")
                        break
        else:
            nnunet_trainer.run_training()

    # Validation (optionally with best)
    if args.val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(args.export_validation_probabilities)


if __name__ == "__main__":
    main()


