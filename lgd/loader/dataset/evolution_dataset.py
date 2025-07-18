import pickle

import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected, degree
import torch 

def compute_node_features(G: nx.Graph, num_nodes: int):
    
    features = []

    # 1. Degree (normalized)
    deg = np.array([G.degree(n) for n in range(num_nodes)])
    deg_norm = deg / (deg.max() + 1e-6)
    features.append(deg_norm[:, None])

    # 2. Laplacian PE (first k eigenvectors)
    L = csgraph.laplacian(nx.adjacency_matrix(G), normed=True)
    eigval, eigvec = eigsh(L, k=8, which='SM')  # small eigs
    features.append(eigvec)

    return np.hstack(features)

def create_train_test_split(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    split_dir = Path(data_dir) / 'split_info'
    split_dir.mkdir(parents=True, exist_ok=True)

    graph_files = sorted(list(data_dir.rglob("*.txt")))
    graph_names = [f.stem for f in graph_files]
    num_graphs = len(graph_names)

    indices = np.arange(num_graphs)
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(train_ratio * num_graphs)
    n_val = int(val_ratio * num_graphs)

    train_names = [graph_names[i] for i in indices[:n_train]]
    val_names = [graph_names[i] for i in indices[n_train:n_train + n_val]]
    test_names = [graph_names[i] for i in indices[n_train + n_val:]]

    # Write to .txt files
    def write_list(names, filename):
        with open(split_dir / filename, 'w') as f:
            for name in names:
                f.write(f"{name}\n")

    write_list(train_names, "train.txt")
    write_list(val_names, "val.txt")
    write_list(test_names, "test.txt")
    return

import torch

def degree_to_bin_fixed(deg_tensor: torch.Tensor,
                        max_degree: int,
                        bin_size: int = 10) -> torch.Tensor:
    """
    Map each degree to an integer bin of equal width (e.g., 1–10 → 1, 11–20 → 2, …).

    Args
    ----
    deg_tensor : shape (N,) or (N, 1) tensor with raw degrees (dtype long / int).
    max_degree : largest degree you expect to encounter.
    bin_size   : width of each bin (default 10).

    Returns
    -------
    bins : tensor of same shape, containing bin indices ∈ {1, 2, …, n_bins}
    """
    # cap degrees that exceed `max_degree`
    deg_capped = torch.clamp(deg_tensor.squeeze(), max=max_degree)

    # bin index starts at 0, then we +1 to keep 0 for padding
    bins = (deg_capped.sub_(1).div(bin_size, rounding_mode='floor')).long() + 1
    return bins.unsqueeze(-1)

class EvolutionDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_dir(self):
        return Path(self.root) / 'raw'
    
    # @property
    # def processed_dir(self):
    #     processed_dir = Path(self.root) / 'processed'
    #     processed_dir.mkdir(parents=True, exist_ok=True)
    #     return processed_dir
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt', 'split_dict.pt']
    
    @property
    def split_info_exists(self):
        """ Return True if split information exists in the raw directory. """
        split_dir = self.root / 'split_info'
        if not split_dir.exists():
            return False
        # Check if any split files exist
        return any(split_dir.iterdir())
    
    def process(self):
        data_list = []
        split_dict = {'train': [], 'valid': [], 'test': []}
        
        if not self.split_info_exists:
            # If split info does not exist, create it
            create_train_test_split(self.root)

        df_file_path = self.raw_dir / 'graph_statistics.csv'
        if not df_file_path.exists():
            raise FileNotFoundError(f"Graph statistics file not found at {df_file_path}. "
                                    "Please ensure the dataset is processed correctly.")
        df_graphs_data = pd.read_csv(df_file_path)

        parse = lambda f: set([x for x in f.read().split('\n')[:-1]])
        split_dir = self.root / 'split_info'
        with open((split_dir / 'train.txt'), 'r') as f:
            train_names = parse(f)
            # assert len(train_names) == 3500
        with open((split_dir / 'val.txt'), 'r') as f:
            val_names = parse(f)
            # assert len(val_names) == 500
        with open((split_dir / 'test.txt'), 'r') as f:
            test_names = parse(f)
            # assert len(test_names) == 1000

        # Assumes raw_dir contains edge list files like graph_001.txt
        file_list = list(self.raw_dir.glob('*.txt'))
        print(len(file_list), "files found in raw directory.")
        
        for file_path in tqdm(file_list):
            graph_name = file_path.stem
            graph_stats = df_graphs_data[df_graphs_data['graph_name'] == graph_name]
            # Load edges
            edge_array = np.loadtxt(file_path, dtype=int)
            edge_index = torch.tensor(edge_array.T, dtype=torch.long)  # shape [2, num_edges]

            # Infer number of nodes
            num_nodes = edge_index.max().item() + 1
            num_edges = edge_index.shape[1]

            # # Compute node features (i.e. degree, Laplacian PE)
            # G = nx.read_edgelist(file_path, nodetype=int)
            # node_features = compute_node_features(G, num_nodes)
            # x = torch.tensor(node_features, dtype=torch.float)

            # Optional: dummy labels for nodes and graphs
            # x = torch.zeros((num_nodes, 1), dtype=torch.long)

            deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
            deg += 1
            deg = degree_to_bin_fixed(deg, 100)  # Shift degree to start from 1
            x = deg.view(-1, 1)
            assert x.min() >= 1, "Degree bins should start from 1."
            edge_attr = torch.zeros((num_edges, 1), dtype=torch.long)
            y = torch.tensor([graph_stats['acc'].values[0]], dtype=torch.float) # TODO: use actual graph labels

            # Convert to undirected graph
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
            data = Data(x=x, 
                        edge_index=edge_index, 
                        y=y, 
                        num_nodes=num_nodes, 
                        # edge_attr=edge_attr, 
                        graph_name=graph_name
                        )
            data_list.append(data)

            ind = len(data_list) - 1
            graph_id = file_path.stem
            if graph_id in train_names:
                split_dict['train'].append(ind)
            elif graph_id in val_names:
                split_dict['valid'].append(ind)
            elif graph_id in test_names:
                split_dict['test'].append(ind)
            else:
                raise ValueError(f'No split assignment for "{graph_id}".')
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])