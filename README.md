# Physics-Informed Deep Flow Prediction

This repository contains a cleaned version of a Stanford CS229
project on airfoil flow-field prediction. The project builds on the
Deep-Flow-Prediction framework from Thuerey et al. and studies lightweight
physics-informed losses for predicting 2D RANS pressure and velocity fields
around airfoils.

Given freestream velocity and an airfoil mask, the model predicts pressure,
horizontal velocity, and vertical velocity on a `128 x 128` Cartesian grid.

## Project Overview

The baseline model is a U-Net-style convolutional encoder-decoder adapted from
the original Deep-Flow-Prediction implementation. This project evaluates three
training objectives:

- Baseline pixelwise L1 loss over pressure and velocity fields.
- Pressure-weighted L1 loss to reduce pressure-channel error.
- Physics-informed continuity loss that penalizes divergence of the predicted
  velocity field in the fluid region.

The final report is available at
[`reports/CS229_Final_Report.pdf`](reports/CS229_Final_Report.pdf).

## Repository Layout

```text
configs/              Experiment YAML files
data/sample/          Small smoke-test dataset
data/README.md        Full dataset instructions
reports/              Final project report
results/figures/      Selected figures for README/report context
scripts/              Training and evaluation entrypoints
src/deep_flow/        Dataset, model, losses, utilities
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version from
https://pytorch.org/get-started/locally/.

## Data

The full reduced dataset contains 6,400 training samples and 90 test samples,
and is not committed to GitHub. Download it from the TUM dataset links listed in
[`data/README.md`](data/README.md), then arrange it as:

```text
data/full/train/*.npz
data/full/test/*.npz
```

Each `.npz` sample contains an array named `a` with six channels:

```text
[u_in_x, u_in_y, airfoil_mask, pressure, u_velocity, v_velocity]
```

The committed `data/sample/` directory is only for smoke tests.

## Quick Start

Run a minimal end-to-end check on the sample data:

```bash
python scripts/train.py --config configs/sample_smoke.yaml --max-iterations 2 --device cpu
python scripts/evaluate.py --config configs/sample_smoke.yaml --checkpoint checkpoints/sample_smoke.pt --device cpu --max-samples 2
```

Train the main experiments after placing the full dataset under `data/full/`:

```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/pressure_weight_2.yaml
python scripts/train.py --config configs/physics_lambda_0.02.yaml
```

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/baseline.pt
```

## Results

Relative error is computed as `sum(abs(prediction - target)) / sum(abs(target))`
on the normalized test fields.

| Model | Pressure Err (%) | Velocity Err (%) | Combined Err (%) |
| --- | ---: | ---: | ---: |
| Baseline | 13.59 | 2.25 | 2.74 |
| Baseline + pressure weight 2 | 12.78 | 2.25 | 2.70 |
| Baseline + pressure weight 4 | 12.09 | 2.28 | 2.73 |
| Physics loss lambda 0.02 | 13.01 | 2.15 | 2.61 |
| Physics loss lambda 0.1 | 12.82 | 2.17 | 2.65 |
| Physics loss lambda 0.5 | 13.86 | 2.36 | 2.86 |
| Physics loss lambda 1.0 | 14.22 | 3.05 | 3.51 |

The moderate physics-informed continuity penalty improves combined error, while
overly strong physics weighting can degrade performance because the resampled
`128 x 128` Cartesian data does not exactly satisfy pointwise continuity.

## Methods

Inputs are normalized freestream velocity grids and an airfoil mask. Targets are
preprocessed by removing pressure offset, masking pressure inside the airfoil,
nondimensionalizing pressure and velocity by freestream magnitude, and scaling
with fixed constants from the original dataset.

The physics-informed objective adds:

```text
L = L_data + lambda_phys * mean((2.04 * du/dx + 2.37 * dv/dy)^2)
```

The divergence term is evaluated only outside the airfoil mask.

## Artifacts

Large artifacts are intentionally excluded from the repository:

- Full `data/train` and `data/test` directories
- Model checkpoints
- Full training/evaluation output folders


## Acknowledgements

This project is based on the open-source Deep-Flow-Prediction code and dataset
from Thuerey, Weissenow, Prantl, and Hu.

```bibtex
@article{thuerey2020deepFlowPred,
  title={Deep learning methods for Reynolds-averaged Navier--Stokes simulations of airfoil flows},
  author={Thuerey, Nils and Wei{\ss}enow, Konstantin and Prantl, Lukas and Hu, Xiangyu},
  journal={AIAA Journal},
  year={2020},
  volume={58},
  number={1},
  pages={25--36},
  publisher={American Institute of Aeronautics and Astronautics}
}
```

## License

The original Deep-Flow-Prediction project is distributed under the license
included in [`LICENSE`](LICENSE). This cleaned project retains that license and
documents additional CS229 project code and experiments.
