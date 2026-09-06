#!/usr/bin/env python
"""Run a trained HRM maze checkpoint on a custom 30x30 maze.

Input format: 30 lines of 30 cells. Use '#', 'S', 'G', and either spaces or
'.' for empty cells. The output uses 'o' for the predicted path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


os.environ.setdefault("DISABLE_COMPILE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from pretrain import PretrainConfig, init_train_state  # noqa: E402


MAZE_CHARSET = "# SGo"
CHAR_TO_ID = {ch: idx + 1 for idx, ch in enumerate(MAZE_CHARSET)}
CHAR_TO_ID["."] = CHAR_TO_ID[" "]


def id_to_char(token_id: int, *, empty_char: str) -> str:
    if token_id == 0:
        return "_"
    if token_id == CHAR_TO_ID[" "]:
        return empty_char
    return MAZE_CHARSET[token_id - 1]


def grid_text(seq: np.ndarray, grid_size: int, *, empty_char: str) -> str:
    arr = seq.reshape(grid_size, grid_size)
    return "\n".join("".join(id_to_char(int(v), empty_char=empty_char) for v in row) for row in arr)


def read_input_text(input_file: str) -> str:
    if input_file == "-":
        return sys.stdin.read()
    return Path(input_file).read_text()


def parse_maze(input_file: str, grid_size: int) -> np.ndarray:
    text = read_input_text(input_file)
    lines = text.splitlines()

    if len(lines) != grid_size:
        raise ValueError(
            f"Expected {grid_size} non-empty maze rows, got {len(lines)}. "
            "Use '.' for empty cells if your editor trims spaces."
        )

    ids: list[int] = []
    for row_idx, line in enumerate(lines, start=1):
        if len(line) != grid_size:
            raise ValueError(
                f"Row {row_idx} has length {len(line)}, expected {grid_size}. "
                "Use '.' for empty cells to preserve trailing empty columns."
            )
        for col_idx, ch in enumerate(line, start=1):
            try:
                ids.append(CHAR_TO_ID[ch])
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported character {ch!r} at row {row_idx}, col {col_idx}. "
                    "Allowed characters are '#', '.', space, 'S', 'G', and 'o'."
                ) from exc

    arr = np.asarray(ids, dtype=np.int32)
    if int((arr == CHAR_TO_ID["S"]).sum()) != 1:
        raise ValueError("Expected exactly one 'S' start cell.")
    if int((arr == CHAR_TO_ID["G"]).sum()) != 1:
        raise ValueError("Expected exactly one 'G' goal cell.")
    return arr


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
    return config, train_state.model, train_metadata


def run_prediction(model, config: PretrainConfig, input_ids: np.ndarray, device: str):
    batch_np = {
        "inputs": input_ids.reshape(1, -1).astype(np.int32),
        "labels": np.full((1, input_ids.size), -100, dtype=np.int32),
        "puzzle_identifiers": np.zeros((1,), dtype=np.int32),
    }
    batch = {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}

    max_steps = int(config.arch.__pydantic_extra__.get("halt_max_steps", 16))  # type: ignore[union-attr]
    predictions: list[np.ndarray] = []
    q_halt_logits: list[float] = []
    q_continue_logits: list[float] = []

    with torch.device(device):
        carry = model.initial_carry(batch)

    with torch.inference_mode():
        for _step in range(max_steps):
            carry, _, _, preds, all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=["logits", "q_halt_logits", "q_continue_logits"],
            )
            predictions.append(preds["logits"].argmax(dim=-1)[0].cpu().numpy().astype(np.int32))
            q_halt_logits.append(float(preds["q_halt_logits"][0].detach().cpu()))
            q_continue_logits.append(float(preds["q_continue_logits"][0].detach().cpu()))

            if bool(all_finish):
                break

    return predictions, q_halt_logits, q_continue_logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Text maze file, or '-' to read from stdin.")
    parser.add_argument("--checkpoint", default="/data/yuanjiayi/HRM/checkpoints/maze-single/step_145831")
    parser.add_argument("--data-path", default="/data/yuanjiayi/HRM/data/maze-30x30-hard-1k")
    parser.add_argument("--output-file", help="Optional path for final prediction text.")
    parser.add_argument("--trace-file", help="Optional path for all recurrent-step predictions.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--space-empty", action="store_true", help="Render empty cells as spaces instead of dots.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the HRM model construction path, but torch.cuda.is_available() is false.")

    checkpoint = Path(args.checkpoint)
    data_path = Path(args.data_path)
    config, model, metadata = load_model(checkpoint, data_path, args.device)

    grid_size = int(round(metadata.seq_len ** 0.5))
    if grid_size * grid_size != metadata.seq_len:
        raise ValueError(f"Expected square maze seq_len, got {metadata.seq_len}")

    input_ids = parse_maze(args.input_file, grid_size)
    predictions, q_halt_logits, q_continue_logits = run_prediction(model, config, input_ids, args.device)

    empty_char = " " if args.space_empty else "."
    final_text = grid_text(predictions[-1], grid_size, empty_char=empty_char)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_text + "\n")
    else:
        print("PREDICTION")
        print(final_text)

    if args.trace_file:
        trace_path = Path(args.trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as f:
            f.write(f"checkpoint: {checkpoint}\n")
            f.write(f"steps_recorded: {len(predictions)}\n\n")
            f.write("INPUT\n")
            f.write(grid_text(input_ids, grid_size, empty_char=empty_char))
            f.write("\n")
            for step_idx, pred in enumerate(predictions, start=1):
                f.write("\n")
                f.write(
                    f"STEP {step_idx:02d} "
                    f"q_halt={q_halt_logits[step_idx - 1]:.4f} "
                    f"q_continue={q_continue_logits[step_idx - 1]:.4f}\n"
                )
                f.write(grid_text(pred, grid_size, empty_char=empty_char))
                f.write("\n")

    print(f"Recorded {len(predictions)} steps.", file=sys.stderr)
    if args.output_file:
        print(f"Wrote final prediction to {args.output_file}", file=sys.stderr)
    if args.trace_file:
        print(f"Wrote recurrent-step trace to {args.trace_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
