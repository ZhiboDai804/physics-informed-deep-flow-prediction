"""Dataset loading and normalization for airfoil flow samples."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_MAX = np.array([100.0, 38.12, 1.0], dtype=np.float32)
TARGET_MAX_DIMLESS = np.array([4.65, 2.04, 2.37], dtype=np.float32)


def _list_npz_files(data_dir, shuffle=False, seed=0, limit=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sorted(data_dir.glob("*.npz"))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(files)
    if limit is not None:
        files = files[: int(limit)]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    return files


def load_raw_samples(data_dir, shuffle=False, seed=0, limit=None):
    """Load compressed numpy samples with shape [N, 6, 128, 128]."""
    files = _list_npz_files(data_dir, shuffle=shuffle, seed=seed, limit=limit)
    samples = np.empty((len(files), 6, 128, 128), dtype=np.float32)
    for idx, file_path in enumerate(files):
        samples[idx] = np.load(file_path)["a"].astype(np.float32)
    return samples, files


def normalize_samples(samples, remove_pressure_offset=True, make_dimless=True):
    """Apply the original Deep-Flow-Prediction normalization pipeline."""
    inputs = samples[:, 0:3].copy()
    targets = samples[:, 3:6].copy()

    if remove_pressure_offset:
        targets[:, 0] -= targets[:, 0].mean(axis=(1, 2), keepdims=True)
        targets[:, 0] *= 1.0 - inputs[:, 2]

    if make_dimless:
        v_norm = np.sqrt(np.max(np.abs(inputs[:, 0]), axis=(1, 2)) ** 2 + np.max(np.abs(inputs[:, 1]), axis=(1, 2)) ** 2)
        v_norm = np.maximum(v_norm, 1e-12)
        targets[:, 0] /= v_norm[:, None, None] ** 2
        targets[:, 1] /= v_norm[:, None, None]
        targets[:, 2] /= v_norm[:, None, None]

    inputs[:, 0] /= INPUT_MAX[0]
    inputs[:, 1] /= INPUT_MAX[1]

    targets[:, 0] /= TARGET_MAX_DIMLESS[0]
    targets[:, 1] /= TARGET_MAX_DIMLESS[1]
    targets[:, 2] /= TARGET_MAX_DIMLESS[2]
    return inputs.astype(np.float32), targets.astype(np.float32)


class AirfoilFlowDataset(Dataset):
    """PyTorch dataset for normalized airfoil flow fields.

    ``split`` can be ``train``, ``validation``, or ``test``. Train and validation
    use a deterministic 80/20 split capped at 400 validation samples, matching
    the original project behavior.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    def __init__(
        self,
        data_dir,
        split="train",
        validation_fraction=0.2,
        max_validation=400,
        shuffle_files=False,
        seed=0,
        limit=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        samples, files = load_raw_samples(self.data_dir, shuffle=shuffle_files, seed=seed, limit=limit)
        inputs, targets = normalize_samples(samples)

        if split in {self.TRAIN, self.VALIDATION}:
            split_index = self._split_index(len(files), validation_fraction, max_validation)
            if split == self.TRAIN:
                inputs, targets, files = inputs[:split_index], targets[:split_index], files[:split_index]
            else:
                inputs, targets, files = inputs[split_index:], targets[split_index:], files[split_index:]
        elif split != self.TEST:
            raise ValueError(f"Unknown split: {split}")

        if len(files) == 0:
            raise ValueError(f"Split {split!r} is empty for {self.data_dir}")

        self.inputs = inputs
        self.targets = targets
        self.files = files

    @staticmethod
    def _split_index(total_length, validation_fraction, max_validation):
        validation_count = min(int(total_length * validation_fraction), max_validation)
        if total_length > 1:
            validation_count = max(1, validation_count)
        return total_length - validation_count

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])

    @staticmethod
    def denormalize(targets, normalized_inputs):
        """Convert normalized targets back to dimensional values.

        ``targets`` and ``normalized_inputs`` are numpy arrays for one sample.
        """
        output = np.array(targets, dtype=np.float32, copy=True)
        output[0] *= TARGET_MAX_DIMLESS[0]
        output[1] *= TARGET_MAX_DIMLESS[1]
        output[2] *= TARGET_MAX_DIMLESS[2]

        ux = normalized_inputs[0] * INPUT_MAX[0]
        uy = normalized_inputs[1] * INPUT_MAX[1]
        v_norm = float(np.sqrt(np.max(np.abs(ux)) ** 2 + np.max(np.abs(uy)) ** 2))
        output[0] *= v_norm**2
        output[1] *= v_norm
        output[2] *= v_norm
        return output
