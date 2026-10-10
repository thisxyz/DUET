from __future__ import annotations

import os

from typing import Optional

import pandas as pd

import torch

from torch_geometric.data import Data

from torch_geometric.utils import subgraph, to_undirected

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

KARATECLUB_URL = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'

def train_val_test_split(num_nodes: int, train_ratio: float=0.5, val_ratio: float=0.25, seed: int=0):

    g = torch.Generator().manual_seed(seed)

    perm = torch.randperm(num_nodes, generator=g)

    n_train = int(num_nodes * train_ratio)

    n_val = int(num_nodes * val_ratio)

    train_idx = perm[:n_train]

    val_idx = perm[n_train:n_train + n_val]

    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True

    val_mask[val_idx] = True

    test_mask[test_idx] = True

    return (train_mask, val_mask, test_mask)

def _scale_features_01(x: torch.Tensor) -> torch.Tensor:

    row_sum = x.sum(dim=1, keepdim=True).clamp(min=1e-08)

    x = x / row_sum

    x_min = x.min(dim=0).values

    x_max = x.max(dim=0).values

    delta = (x_max - x_min).clamp(min=1e-08)

    return (x - x_min) / delta

def _load_cora_ucsc(raw_dir: str) -> Data:

    content_path = os.path.join(raw_dir, 'cora.content')

    cites_path = os.path.join(raw_dir, 'cora.cites')

    idx_features_labels = []

    with open(content_path, 'r') as f:

        for line in f:

            parts = line.strip().split()

            idx_features_labels.append(parts)

    paper_ids = [row[0] for row in idx_features_labels]

    id_map = {pid: i for i, pid in enumerate(paper_ids)}

    features = torch.tensor([[float(v) for v in row[1:-1]] for row in idx_features_labels], dtype=torch.float)

    labels_raw = [row[-1] for row in idx_features_labels]

    classes = sorted(set(labels_raw))

    class_map = {c: i for i, c in enumerate(classes)}

    y = torch.tensor([class_map[c] for c in labels_raw], dtype=torch.long)

    edges = []

    with open(cites_path, 'r') as f:

        for line in f:

            a, b = line.strip().split()

            if a in id_map and b in id_map:

                i, j = (id_map[a], id_map[b])

                edges.append((i, j))

                edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_index = torch.unique(edge_index, dim=1)

    data = Data(x=features, edge_index=edge_index, y=y, num_nodes=features.size(0))

    data.num_features = features.size(1)

    data.num_classes = len(classes)

    return data

def _finalize(data: Data, seed: int) -> Data:

    data = data.clone()

    data.x = _scale_features_01(data.x.float())

    data.y = data.y.long()

    data.x_range = (0.0, 1.0)

    data.train_mask, data.val_mask, data.test_mask = train_val_test_split(data.num_nodes, seed=seed)

    if not hasattr(data, 'num_features') or data.num_features is None:

        data.num_features = data.x.size(1)

    if not hasattr(data, 'num_classes') or data.num_classes is None:

        data.num_classes = int(data.y.max().item()) + 1

    return data

def load_cora(root: str=DATA_ROOT, seed: int=0, raw_dir: Optional[str]=None):

    if raw_dir is None:

        raw_dir = os.path.join(root, 'cora_raw')

    cache = os.path.join(root, 'cora_processed.pt')

    if os.path.exists(cache):

        data = torch.load(cache, weights_only=False)

    else:

        if os.path.exists(os.path.join(raw_dir, 'cora.content')):

            data = _load_cora_ucsc(raw_dir)

        else:

            from torch_geometric.datasets import Planetoid

            os.makedirs(root, exist_ok=True)

            dataset = Planetoid(root=os.path.join(root, 'Planetoid'), name='Cora')

            data = dataset[0]

            data.num_features = dataset.num_features

            data.num_classes = dataset.num_classes

        os.makedirs(root, exist_ok=True)

        torch.save(data, cache)

    return _finalize(data, seed)

def load_citeseer(root: str=DATA_ROOT, seed: int=0):

    from torch_geometric.datasets import Planetoid

    cache = os.path.join(root, 'citeseer_processed.pt')

    if os.path.exists(cache):

        data = torch.load(cache, weights_only=False)

    else:

        dataset = Planetoid(root=os.path.join(root, 'Planetoid'), name='CiteSeer')

        data = dataset[0]

        data.num_features = dataset.num_features

        data.num_classes = dataset.num_classes

        os.makedirs(root, exist_ok=True)

        torch.save(data, cache)

    return _finalize(data, seed)

def load_pubmed(root: str=DATA_ROOT, seed: int=0):

    from torch_geometric.datasets import Planetoid

    cache = os.path.join(root, 'pubmed_processed.pt')

    if os.path.exists(cache):

        data = torch.load(cache, weights_only=False)

    else:

        dataset = Planetoid(root=os.path.join(root, 'Planetoid'), name='PubMed')

        data = dataset[0]

        data.num_features = dataset.num_features

        data.num_classes = dataset.num_classes

        os.makedirs(root, exist_ok=True)

        torch.save(data, cache)

    return _finalize(data, seed)

def _filter_top_class(data: Data, num_classes: int) -> Data:

    y_oh = torch.nn.functional.one_hot(data.y.long())

    counts = y_oh.sum(dim=0)

    keep_labels = counts.sort(descending=True).indices[:num_classes]

    y_keep = y_oh[:, keep_labels]

    idx = y_keep.sum(dim=1).bool()

    data = data.clone()

    data.x = data.x[idx]

    data.y = y_keep[idx].argmax(dim=1)

    data.num_nodes = int(data.y.size(0))

    data.edge_index, _ = subgraph(idx, data.edge_index, relabel_nodes=True, num_nodes=idx.numel())

    data.num_features = data.x.size(1)

    data.num_classes = int(data.y.max().item()) + 1

    return data

def _download_karateclub_csv(name: str, raw_dir: str):

    import urllib.request

    os.makedirs(raw_dir, exist_ok=True)

    for part in ('edges', 'features', 'target'):

        dest = os.path.join(raw_dir, f'{part}.csv')

        if os.path.isfile(dest) and os.path.getsize(dest) > 1000:

            continue

        url = f'{KARATECLUB_URL}/{name}/{part}.csv'

        mirrors = [url, f'https://ghproxy.net/{url}', f'https://cdn.jsdelivr.net/gh/benedekrozemberczki/karateclub@master/dataset/node_level/{name}/{part}.csv']

        last_err = None

        for u in mirrors:

            try:

                urllib.request.urlretrieve(u, dest)

                if os.path.getsize(dest) > 1000:

                    break

            except Exception as e:

                last_err = e

        else:

            raise RuntimeError(f'Failed to download {name}/{part}.csv: {last_err}')

def _load_karateclub(name: str, root: str) -> Data:

    raw_dir = os.path.join(root, 'karateclub', name, 'raw')

    _download_karateclub_csv(name, raw_dir)

    y = torch.from_numpy(pd.read_csv(os.path.join(raw_dir, 'target.csv'))['target'].to_numpy(dtype=int).copy())

    num_nodes = len(y)

    edge_index = torch.from_numpy(pd.read_csv(os.path.join(raw_dir, 'edges.csv')).to_numpy().copy()).t().contiguous()

    edge_index = to_undirected(edge_index, num_nodes)

    x = pd.read_csv(os.path.join(raw_dir, 'features.csv')).drop_duplicates()

    x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)

    x = x.reindex(range(num_nodes), fill_value=0)

    x = torch.from_numpy(x.to_numpy().astype('float32')).float()

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    data.num_features = data.x.size(1)

    data.num_classes = int(data.y.max().item()) + 1

    return data

def load_lastfm(root: str=DATA_ROOT, seed: int=0):

    cache = os.path.join(root, 'lastfm_lpgnn_processed.pt')

    if os.path.exists(cache):

        data = torch.load(cache, weights_only=False)

    else:

        data = _load_karateclub('lastfm', root)

        data = _filter_top_class(data, num_classes=10)

        os.makedirs(root, exist_ok=True)

        torch.save(data, cache)

    return _finalize(data, seed)

def load_facebook(root: str=DATA_ROOT, seed: int=0):

    cache = os.path.join(root, 'facebook_lpgnn_processed.pt')

    if os.path.exists(cache):

        data = torch.load(cache, weights_only=False)

    else:

        data = _load_karateclub('facebook', root)

        os.makedirs(root, exist_ok=True)

        torch.save(data, cache)

    return _finalize(data, seed)

LOADERS = {'cora': load_cora, 'citeseer': load_citeseer, 'pubmed': load_pubmed, 'lastfm': load_lastfm, 'facebook': load_facebook}

def load_dataset(name: str, seed: int=0, root: str=DATA_ROOT):

    key = name.lower()

    if key not in LOADERS:

        raise ValueError(f'Unknown dataset {name}; choose from {list(LOADERS)}')

    return LOADERS[key](root=root, seed=seed)

def edge_index_to_dense(edge_index: torch.Tensor, num_nodes: int, device=None) -> torch.Tensor:

    adj = torch.zeros(num_nodes, num_nodes, device=device or edge_index.device)

    adj[edge_index[0], edge_index[1]] = 1.0

    adj.fill_diagonal_(0.0)

    adj = torch.maximum(adj, adj.t())

    return adj
