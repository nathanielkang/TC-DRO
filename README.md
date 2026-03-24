# Taming the Tails: DRO for Imbalanced Regression

Target-Conditional Distributionally Robust Optimization (TC-DRO) for imbalanced regression. This framework estimates local sample density in target space to define target-conditional ambiguity sets, solves a dual DRO problem with adaptive sample reweighting, and provides worst-case performance guarantees on rare target regions.

## Features

- Target-conditional ambiguity sets (chi-squared and Wasserstein variants)
- Adaptive sample reweighting based on target rarity
- Multiple base learners: MLP, XGBoost, linear
- Imbalanced regression metrics: SERA, tail MSE/MAE, coverage

## Installation

```bash
cd 2_Code_a
pip install -r requirements.txt
```

## Project Structure

```
2_Code_a/
├── configs/
│   └── default.yaml      # Hyperparameters
├── src/
│   ├── data/
│   │   └── datasets.py   # Data loading, relevance, density estimation
│   ├── dro/
│   │   ├── ambiguity_sets.py   # TC ambiguity sets
│   │   └── tc_dro_trainer.py   # Training loop
│   ├── evaluation/
│   │   └── metrics.py    # MSE, MAE, SERA, tail metrics
│   └── models/
│       └── base_models.py # MLP, XGBoost, linear
├── scripts/
│   ├── run_experiments.py
│   └── smoke_test.py
├── requirements.txt
└── README.md
```

## Running the Smoke Test

Verify the code skeleton and environment:

```bash
python scripts/smoke_test.py
```

Expected output: `PASS: All smoke tests passed.`

## Running Experiments

```bash
python scripts/run_experiments.py --config configs/default.yaml --output results/
```

(Requires implementation of stubs.)

## Supported Datasets

abalone, ailerons, accel, bank8FM, cpu_act, delta_ailerons, machine_cpu, airfoil

## License

Research use.
