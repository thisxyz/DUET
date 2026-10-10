# DUET

Graph-structured data are widely used in real-world applications, yet both node features and graph links may contain sensitive information, raising significant privacy concerns in graph learning. Local differential privacy (LDP) provides a promising solution by allowing users to perturb sensitive data locally before sharing them with an untrusted server. However, most existing locally differentially private graph learning methods protect only a single graph view, leaving either node features or graph links exposed to potential privacy leakage. Although a few dual-view solutions protect both views, they typically process privatized node features and graph links largely independently, overlooking their intrinsic dependencies and consequently suffering from limited learning utility. To address these limitations, we propose DUET, a collaborative utility recovery protocol for locally differentially private graph learning. DUET follows a protect independently, recover collaboratively principle: node features and graph links are independently privatized on decentralized clients, while their cross-view dependencies are exploited on the server to recover learning utility without incurring additional privacy loss. Specifically, DUET consists of four key components: local graph perturbation, private feature refinement, private link reconstruction, and cross-view collaborative recovery. Extensive experiments across multiple graph datasets, GNN backbones, and LDP mechanisms demonstrate that DUET consistently improves graph learning utility under rigorous dual-view local privacy protection.

<img width="2188" height="514" alt="image" src="https://github.com/user-attachments/assets/7a5838d6-3a13-4452-a7cf-2a4e8e41b006" />

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
