"""Evaluate a trained checkpoint on an airfoil flow test set."""

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_flow.dataset import AirfoilFlowDataset
from deep_flow.model import TurbNetG
from deep_flow.utils import ensure_dir, load_config, resolve_device, save_json
from deep_flow.visualization import save_prediction_grid


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative_error(prediction, target):
    numerator = np.sum(np.abs(prediction - target))
    denominator = max(float(np.sum(np.abs(target))), 1e-12)
    return float(numerator / denominator)


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint to evaluate.")
    parser.add_argument("--output-dir", default=None, help="Directory for metrics and figures.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for quick checks.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, or cuda.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device or config["training"].get("device", "auto"))

    dataset = AirfoilFlowDataset(
        project_path(config["data"]["test_dir"]),
        split=AirfoilFlowDataset.TEST,
        limit=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = TurbNetG(
        channel_exponent=float(config["model"].get("channel_exponent", 5)),
        dropout=float(config["model"].get("dropout", 0.0)),
    )
    checkpoint = torch.load(project_path(args.checkpoint), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    output_dir = project_path(args.output_dir or Path(config["output"]["dir"]) / f"{config['experiment_name']}_eval")
    figure_dir = output_dir / "figures"
    ensure_dir(figure_dir)

    pressure_errors = []
    velocity_errors = []
    combined_errors = []
    for idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        prediction = model(inputs)

        pred_np = prediction.cpu().numpy()[0]
        target_np = targets.cpu().numpy()[0]
        pressure_errors.append(relative_error(pred_np[0], target_np[0]))
        velocity_errors.append(relative_error(pred_np[1:3], target_np[1:3]))
        combined_errors.append(relative_error(pred_np, target_np))

        if idx < int(config["output"].get("num_eval_figures", 5)):
            save_prediction_grid(figure_dir / f"sample_{idx:04d}.png", pred_np, target_np, title=dataset.files[idx].stem)

    metrics = {
        "num_samples": len(dataset),
        "pressure_error_percent": float(np.mean(pressure_errors) * 100.0),
        "velocity_error_percent": float(np.mean(velocity_errors) * 100.0),
        "combined_error_percent": float(np.mean(combined_errors) * 100.0),
    }
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
