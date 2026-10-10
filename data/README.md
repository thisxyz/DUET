# Data

- `cora_raw/`: Cora content/cites files
- `karateclub/lastfm/raw/`: LastFM edges, features, target (LPGNN)
- `karateclub/facebook/raw/`: Facebook page-page network (LPGNN)

CiteSeer is downloaded automatically via `torch_geometric.datasets.Planetoid` on first use.
Processed caches (`*_processed.pt`) are created under this folder at runtime.
