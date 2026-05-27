"""Loss functions for baseline, pressure-weighted, and physics-informed runs."""

import torch
import torch.nn as nn


def channel_l1_loss(prediction, target, pressure_weight=1.0, velocity_weight=1.0):
    """Weighted L1 loss over pressure and velocity channels."""
    criterion = nn.L1Loss()
    pressure_loss = criterion(prediction[:, 0:1], target[:, 0:1])
    velocity_loss = criterion(prediction[:, 1:3], target[:, 1:3])
    total = pressure_weight * pressure_loss + velocity_weight * velocity_loss
    return total, {
        "loss_pressure": pressure_loss.detach(),
        "loss_velocity": velocity_loss.detach(),
    }


def _gradient_x(tensor):
    grad = torch.zeros_like(tensor)
    grad[:, :, :, 1:-1] = 0.5 * (tensor[:, :, :, 2:] - tensor[:, :, :, :-2])
    grad[:, :, :, 0] = tensor[:, :, :, 1] - tensor[:, :, :, 0]
    grad[:, :, :, -1] = tensor[:, :, :, -1] - tensor[:, :, :, -2]
    return grad


def _gradient_y(tensor):
    grad = torch.zeros_like(tensor)
    grad[:, :, 1:-1, :] = 0.5 * (tensor[:, :, 2:, :] - tensor[:, :, :-2, :])
    grad[:, :, 0, :] = tensor[:, :, 1, :] - tensor[:, :, 0, :]
    grad[:, :, -1, :] = tensor[:, :, -1, :] - tensor[:, :, -2, :]
    return grad


def divergence_loss(prediction, inputs, scale_u=2.04, scale_v=2.37, reduction="mean"):
    """Continuity residual loss on predicted normalized velocity fields."""
    u = prediction[:, 1:2]
    v = prediction[:, 2:3]
    mask = inputs[:, 2:3]

    divergence = scale_u * _gradient_x(u) + scale_v * _gradient_y(v)
    fluid_weight = 1.0 - mask
    residual = divergence.pow(2) * fluid_weight
    if reduction == "mean":
        return residual.mean()
    if reduction == "none":
        return residual
    raise ValueError(f"Unsupported reduction: {reduction}")


def compute_training_loss(prediction, target, inputs, loss_config):
    """Compute total loss from a config dictionary."""
    loss_type = loss_config.get("type", "baseline")
    pressure_weight = float(loss_config.get("pressure_weight", 1.0))
    velocity_weight = float(loss_config.get("velocity_weight", 1.0))
    physics_weight = float(loss_config.get("physics_weight", 0.0))

    data_loss, metrics = channel_l1_loss(
        prediction,
        target,
        pressure_weight=pressure_weight,
        velocity_weight=velocity_weight,
    )
    physics = prediction.new_tensor(0.0)
    if loss_type == "physics" or physics_weight > 0.0:
        physics = divergence_loss(prediction, inputs)

    total = data_loss + physics_weight * physics
    metrics.update(
        {
            "loss_data": data_loss.detach(),
            "loss_physics": physics.detach(),
            "loss_total": total.detach(),
        }
    )
    return total, metrics
