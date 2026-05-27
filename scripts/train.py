"""Train an airfoil flow prediction model from a YAML config."""

from argparse import ArgumentParser
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from deep_flow.dataset import AirfoilFlowDataset
from deep_flow.losses import compute_training_loss
from deep_flow.model import TurbNetG, weights_init
from deep_flow.utils import AverageMeter, compute_lr, ensure_dir, load_config, resolve_device, save_json, set_seed
from deep_flow.visualization import save_prediction_grid


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_loader(config, split):
    data_cfg = config["data"]
    data_dir = data_cfg["train_dir"] if split != AirfoilFlowDataset.TEST else data_cfg["test_dir"]
    dataset = AirfoilFlowDataset(
        project_path(data_dir),
        split=split,
        validation_fraction=float(data_cfg.get("validation_fraction", 0.2)),
        max_validation=int(data_cfg.get("max_validation", 400)),
        shuffle_files=bool(data_cfg.get("shuffle_files", False)) and split == AirfoilFlowDataset.TRAIN,
        seed=int(config.get("seed", 0)),
        limit=data_cfg.get(f"{split}_limit"),
    )
    batch_size = int(config["training"]["batch_size"])
    drop_last = split == AirfoilFlowDataset.TRAIN and len(dataset) >= batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == AirfoilFlowDataset.TRAIN,
        drop_last=drop_last,
    )


def run_epoch(model, loader, optimizer, device, loss_config):
    model.train()
    meters = {
        "loss_total": AverageMeter(),
        "loss_data": AverageMeter(),
        "loss_physics": AverageMeter(),
        "loss_pressure": AverageMeter(),
        "loss_velocity": AverageMeter(),
    }
    for inputs, targets in loader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss, metrics = compute_training_loss(prediction, targets, inputs, loss_config)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        for name, value in metrics.items():
            meters[name].update(value.item(), batch_size)
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def validate(model, loader, device, loss_config, figure_path=None):
    model.eval()
    meters = {
        "val_loss_total": AverageMeter(),
        "val_loss_data": AverageMeter(),
        "val_loss_physics": AverageMeter(),
        "val_loss_pressure": AverageMeter(),
        "val_loss_velocity": AverageMeter(),
    }
    first_batch = None
    for inputs, targets in loader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        prediction = model(inputs)
        _, metrics = compute_training_loss(prediction, targets, inputs, loss_config)
        batch_size = inputs.shape[0]
        for name, value in metrics.items():
            meters[f"val_{name}"].update(value.item(), batch_size)
        if first_batch is None:
            first_batch = (inputs.cpu(), targets.cpu(), prediction.cpu())

    if figure_path and first_batch is not None:
        inputs, targets, prediction = first_batch
        save_prediction_grid(figure_path, prediction[0].numpy(), targets[0].numpy(), title="Validation sample")

    return {name: meter.average for name, meter in meters.items()}


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override training iterations for smoke tests.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, or cuda.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 0)))
    device = resolve_device(args.device or config["training"].get("device", "auto"))

    train_loader = build_loader(config, AirfoilFlowDataset.TRAIN)
    val_loader = build_loader(config, AirfoilFlowDataset.VALIDATION)
    iterations = int(args.max_iterations or config["training"]["iterations"])
    epochs = max(1, int(math.ceil(iterations / max(len(train_loader), 1))))

    model_cfg = config["model"]
    model = TurbNetG(
        channel_exponent=float(model_cfg.get("channel_exponent", 5)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.apply(weights_init)
    model.to(device)

    training_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        betas=tuple(training_cfg.get("betas", [0.5, 0.999])),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    output_dir = project_path(config["output"]["dir"]) / config["experiment_name"]
    checkpoint_dir = project_path(config["output"].get("checkpoint_dir", "checkpoints"))
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)

    history = []
    for epoch in range(epochs):
        if training_cfg.get("decay_lr", True):
            lr = compute_lr(
                epoch,
                epochs,
                float(training_cfg["learning_rate"]) * 0.1,
                float(training_cfg["learning_rate"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

        train_metrics = run_epoch(model, train_loader, optimizer, device, config["loss"])
        figure_path = output_dir / f"validation_epoch_{epoch:04d}.png" if epoch == epochs - 1 else None
        val_metrics = validate(model, val_loader, device, config["loss"], figure_path=figure_path)
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **train_metrics, **val_metrics}
        history.append(record)
        print(record)

    checkpoint_path = checkpoint_dir / f"{config['experiment_name']}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
        },
        checkpoint_path,
    )
    save_json(output_dir / "history.json", history)
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
