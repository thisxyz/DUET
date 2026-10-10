# DUET

This repository contains the core implementation of **DUET**, including dual-view LDP mechanisms, private feature refinement, private link reconstruction, structure-aware feature recovery (SAFR), feature-assisted edge calibration, and GNN training utilities.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric
- pandas, numpy, scipy, openpyxl

```bash
pip install -r requirements.txt
```

## Datasets

Bundled under `data/`:

| Dataset | Source | Notes |
|---------|--------|-------|
| Cora | `data/cora_raw/` | Citation network |
| CiteSeer | auto-download via Planetoid | Cached on first run |
| LastFM | `data/karateclub/lastfm/` | LPGNN setting, FilterTopClass(10) |
| Facebook | `data/karateclub/facebook/` | LPGNN KarateClub page-page network |

Default split: random 50% / 25% / 25% (train / val / test).

## Quick Start

```bash
python examples/run_cora.py
```

This runs Vanilla vs DUET on Cora with GCN under the default privacy budgets:

- Feature LDP: Multi-Bit Mechanism (MBM), `ε_X = 1.0`
- Link LDP: Randomized Response (RR), `ε_A = 8.0`

## Project Layout

```
DUET/
├── duet/
│   ├── mechanisms/     # MBM / PM / SWM, RR
│   ├── recovery/       # FR, Bayesian link recovery, SAFR, edge calibration
│   ├── models/         # Dense GCN/GAT/SAGE, DUET fusion model
│   ├── baselines/      # Dual-view adaptations of compared methods
│   ├── data.py         # Dataset loaders
│   ├── methods.py      # Method runners (Vanilla, DUET, baselines)
│   ├── attacks.py      # Feature / link inference attacks (AUC)
│   └── train.py        # Training utilities
├── examples/
│   └── run_cora.py
├── data/
└── requirements.txt
```

## Default Experimental Setting

Aligned with the paper's ablation / parameter / privacy analyses:

- Backbone: GCN (also supports GAT / GraphSAGE)
- Privacy: MBM + RR, `ε_X=1.0`, `ε_A=8.0`
- Datasets: Cora, CiteSeer, LastFM, Facebook

## License

For academic research use. Please refer to the paper for experimental protocols and threat models.
