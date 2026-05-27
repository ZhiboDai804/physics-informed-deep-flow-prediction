# Data

The full training and test data are not committed to this repository because
the reduced dataset is approximately 1.2 GB.

Expected local layout for full experiments:

```text
data/full/train/*.npz
data/full/test/*.npz
```

Each `.npz` file stores an array named `a` with shape `[6, 128, 128]`:

```text
[u_in_x, u_in_y, airfoil_mask, pressure, u_velocity, v_velocity]
```

The original data source is the Deep-Flow-Prediction dataset released by
Thuerey et al.:

- Reduced 6.4k sample dataset: https://dataserv.ub.tum.de/s/m1470791
- Full 53.8k sample dataset: https://dataserv.ub.tum.de/s/m1459172

The `data/sample/` directory contains a few small examples copied from the
reduced dataset for smoke tests only.
