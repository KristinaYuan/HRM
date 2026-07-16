#!/usr/bin/env python
"""Dump step-by-step maze predictions for an HRM checkpoint.

This script visualizes the model's recurrent evaluation trajectory. It does
not expose natural-language reasoning; it records the answer grid predicted
after each ACT step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


os.environ.setdefault("DISABLE_COMPILE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from pretrain import PretrainConfig, init_train_state  # noqa: E402


ID_TO_CHAR = {
    -100: ".",
    0: "_",
    1: "#",
    2: " ",
    3: "S",
    4: "G",
    5: "o",
}


def parse_indices(raw: str | None, *, start: int, num_samples: int) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return list(range(start, start + num_samples))


def grid_text(seq: np.ndarray, grid_size: int) -> str:
    arr = seq.reshape(grid_size, grid_size)
    return "\n".join("".join(ID_TO_CHAR.get(int(v), "?") for v in row) for row in arr)


def accuracy(pred: np.ndarray, labels: np.ndarray) -> tuple[float, bool, int]:
    mask = labels != -100
    correct = (pred == labels) & mask
    total = int(mask.sum())
    errors = int(total - correct.sum())
    return float(correct.sum() / max(total, 1)), errors == 0, errors


def path_stats(pred: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    pred_path = pred == 5
    true_path = labels == 5
    tp = int((pred_path & true_path).sum())
    fp = int((pred_path & ~true_path).sum())
    fn = int((~pred_path & true_path).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "path_tp": tp,
        "path_fp": fp,
        "path_fn": fn,
        "path_precision": precision,
        "path_recall": recall,
    }


def example_to_puzzle_ids(puzzle_indices: np.ndarray, puzzle_identifiers: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    puzzle_ids = []
    for idx in indices:
        puzzle_idx = int(np.searchsorted(puzzle_indices, idx, side="right") - 1)
        puzzle_ids.append(int(puzzle_identifiers[puzzle_idx]))
    return np.asarray(puzzle_ids, dtype=np.int32)


def load_model(checkpoint: Path, data_path: Path, device: str):
    with open(checkpoint.parent / "all_config.yaml", "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    config.data_path = str(data_path)

    with open(data_path / "train" / "dataset.json", "r") as f:
        train_metadata = PuzzleDatasetMetadata(**json.load(f))

    train_state = init_train_state(config, train_metadata, world_size=1)
    state_dict = torch.load(checkpoint, map_location=device)
    try:
        train_state.model.load_state_dict(state_dict, assign=True)
    except RuntimeError:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}, assign=True)

    train_state.model.eval()
    return config, train_state.model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/data/yuanjiayi/HRM/checkpoints/maze-single/step_145831")
    parser.add_argument("--data-path", default="/data/yuanjiayi/HRM/data/maze-30x30-hard-1k")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--indices", help="Comma-separated example indices, e.g. 0,17,42")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", default="/data/yuanjiayi/HRM/inspections/maze-step_145831")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the HRM model construction path, but torch.cuda.is_available() is false.")

    config, model = load_model(checkpoint, data_path, args.device)
    sample_indices = parse_indices(args.indices, start=args.start, num_samples=args.num_samples)

    split_dir = data_path / args.split
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy").astype(np.int32)
    puzzle_indices = np.load(split_dir / "all__puzzle_indices.npy")
    puzzle_identifiers = np.load(split_dir / "all__puzzle_identifiers.npy")
    with open(split_dir / "dataset.json", "r") as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))

    if metadata.ignore_label_id is not None:
        labels[labels == metadata.ignore_label_id] = -100

    grid_size = int(round(metadata.seq_len ** 0.5))
    if grid_size * grid_size != metadata.seq_len:
        raise ValueError(f"Expected square maze seq_len, got {metadata.seq_len}")

    batch_np = {
        "inputs": inputs[sample_indices].astype(np.int32),
        "labels": labels[sample_indices].astype(np.int32),
        "puzzle_identifiers": example_to_puzzle_ids(puzzle_indices, puzzle_identifiers, sample_indices),
    }
    batch = {k: torch.from_numpy(v).to(args.device) for k, v in batch_np.items()}

    traces: list[dict] = []
    step_predictions: list[np.ndarray] = []
    q_halt_logits: list[np.ndarray] = []
    q_continue_logits: list[np.ndarray] = []

    with torch.device(args.device):
        carry = model.initial_carry(batch)
    max_steps = int(config.arch.__pydantic_extra__.get("halt_max_steps", 16))  # type: ignore[union-attr]

    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            carry, _, _, preds, all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=["logits", "q_halt_logits", "q_continue_logits"],
            )
            pred_ids = preds["logits"].argmax(dim=-1).cpu().numpy()
            step_predictions.append(pred_ids)
            q_halt_logits.append(preds["q_halt_logits"].cpu().numpy())
            q_continue_logits.append(preds["q_continue_logits"].cpu().numpy())

            step_summary = {"step": step, "samples": []}
            for row, sample_idx in enumerate(sample_indices):
                acc, exact, errors = accuracy(pred_ids[row], batch_np["labels"][row])
                sample_stats = {
                    "index": sample_idx,
                    "accuracy": acc,
                    "exact": exact,
                    "errors": errors,
                    "q_halt_logit": float(q_halt_logits[-1][row]),
                    "q_continue_logit": float(q_continue_logits[-1][row]),
                    **path_stats(pred_ids[row], batch_np["labels"][row]),
                }
                step_summary["samples"].append(sample_stats)
            traces.append(step_summary)

            if bool(all_finish):
                break

    step_predictions_np = np.stack(step_predictions, axis=0)
    q_halt_np = np.stack(q_halt_logits, axis=0)
    q_continue_np = np.stack(q_continue_logits, axis=0)

    np.save(output_dir / "step_predictions.npy", step_predictions_np)
    np.save(output_dir / "q_halt_logits.npy", q_halt_np)
    np.save(output_dir / "q_continue_logits.npy", q_continue_np)
    np.save(output_dir / "inputs.npy", batch_np["inputs"])
    np.save(output_dir / "labels.npy", batch_np["labels"])

    summary = {
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "split": args.split,
        "sample_indices": sample_indices,
        "grid_size": grid_size,
        "steps_recorded": int(step_predictions_np.shape[0]),
        "token_map": {str(k): v for k, v in ID_TO_CHAR.items()},
        "trace": traces,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for row, sample_idx in enumerate(sample_indices):
        with open(output_dir / f"sample_{sample_idx:04d}_trace.txt", "w") as f:
            f.write(f"checkpoint: {checkpoint}\n")
            f.write(f"split/index: {args.split}/{sample_idx}\n")
            f.write(f"token map: {ID_TO_CHAR}\n\n")
            f.write("INPUT\n")
            f.write(grid_text(batch_np["inputs"][row], grid_size))
            f.write("\n\nTARGET\n")
            f.write(grid_text(batch_np["labels"][row], grid_size))
            f.write("\n")

            for step_id, pred_grid in enumerate(step_predictions_np[:, row], start=1):
                sample_stats = traces[step_id - 1]["samples"][row]
                f.write("\n")
                f.write(
                    f"STEP {step_id:02d} "
                    f"acc={sample_stats['accuracy']:.4f} "
                    f"exact={sample_stats['exact']} "
                    f"errors={sample_stats['errors']} "
                    f"q_halt={sample_stats['q_halt_logit']:.4f} "
                    f"q_continue={sample_stats['q_continue_logit']:.4f}\n"
                )
                f.write(grid_text(pred_grid, grid_size))
                f.write("\n")

    print(f"Wrote maze reasoning trace to {output_dir}")
    print(f"Recorded {step_predictions_np.shape[0]} steps for samples: {sample_indices}")


if __name__ == "__main__":
    main()
