# Dataset Instructions — Tennessee Eastman Process (TEP)

This project uses the **Tennessee Eastman Process (TEP)** benchmark dataset, a standard benchmark for fault detection and diagnosis research in industrial process control.

## Download

The dataset is publicly available on Kaggle:

👉 [Tennessee Eastman Process Simulation Dataset — Kaggle](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset)

## Expected folder structure after download

Place the files in the `data/TEdata/` folder as follows:

```
data/
└── TEdata/
    ├── X       # Normal operating condition (training)
    ├── Xv      # Validation/test dataset
    ├── Xt1     # Fault scenario IDV1
    ├── Xt2     # Fault scenario IDV2
    ├── ...
    └── Xt21    # Fault scenario IDV21
```

## Dataset description

| File | Description |
|---|---|
| `X` | Normal operating data — used to train PCA/DAE-PCA baseline models |
| `Xv` | Validation/test data under normal conditions |
| `Xt1`–`Xt21` | 21 fault scenarios — each file contains sensor measurements under a specific TEP fault case |

## Original reference

> J. J. Downs and E. F. Vogel, "A plant-wide industrial process control problem,"
> *Computers & Chemical Engineering*, vol. 17, no. 3, pp. 245–255, 1993.
