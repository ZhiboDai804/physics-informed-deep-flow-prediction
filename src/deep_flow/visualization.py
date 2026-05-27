"""Visualization helpers for model predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_prediction_grid(path, prediction, target, title=None):
    """Save a 3x3 grid: prediction, ground truth, absolute error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    outputs = np.array(prediction, copy=True)
    targets = np.array(target, copy=True)
    for channel in range(3):
        outputs[channel] = np.flipud(outputs[channel].transpose())
        targets[channel] = np.flipud(targets[channel].transpose())

    channel_names = ["pressure", "u-velocity", "v-velocity"]
    row_labels = ["Prediction", "Ground truth", "Error"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for channel in range(3):
        vmin = min(float(outputs[channel].min()), float(targets[channel].min()))
        vmax = max(float(outputs[channel].max()), float(targets[channel].max()))
        plots = [
            outputs[channel],
            targets[channel],
            np.abs(outputs[channel] - targets[channel]),
        ]
        for row, values in enumerate(plots):
            ax = axes[row, channel]
            kwargs = {"cmap": "magma"}
            if row < 2:
                kwargs.update({"vmin": vmin, "vmax": vmax})
            image = ax.imshow(values, origin="lower", **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(channel_names[channel])
            if channel == 0:
                ax.set_ylabel(row_labels[row])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title or path.stem, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)
