import os
import numpy as np
import argparse
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset


def take_second(element):
    return element[1]


def temp_partition_arxiv(data, year_bound, proportion=1.0):
    """
    Temporally partition ogbn-arxiv data based on year boundaries
    :param data: torch_geometric.data.Data, the ogbn-arxiv data
    :param year_bound: List[int] with length = 3, years for training: year_bound[0] to year_bound[1]-1, years for eval: year_bound[1] to year_bound[2] - 1
    :param proportion: float, percentage of nodes to be kept based on node degree
    :return: torch_geometric.data.Data, the partitioned ogbn-arxiv data
    """

    assert len(year_bound) == 3
    node_years = data['node_year']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years)) # frequency of interaction of each node before year upper bound
    edges = data["edge_index"]
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] < year_bound[2] and node_years[edges[1][i]] < year_bound[2]: # if the edge happens before year upper bound
            d[edges[0][i]] += 1
            d[edges[1][i]] += 1

    nodes = [] # node id and frequency of interaction before year upper bound
    for i, year in enumerate(node_years):
        if year < year_bound[2]:
            nodes.append([i, d[i]])

    nodes.sort(key=take_second, reverse=True)

    nodes_new = nodes[:int(proportion * len(nodes))] # take top popular nodes that happens before year upper bound

    # reproduce id
    result_edges = []
    result_features = []
    for node in nodes_new:
        result_features.append(data.x[node[0]])
    result_features = torch.stack(result_features)

    ids = {}
    for i, node in enumerate(nodes_new):
        ids[node[0]] = i # previous node id to new node id

    for i in range(edges.shape[1]):
        if edges[0][i].item() in ids and edges[1][i].item() in ids: # if in node and out node of an edge are both in result nodes, add the edge
            result_edges.append([ids[edges[0][i].item()], ids[edges[1][i].item()]])
    result_edges = torch.LongTensor(result_edges).transpose(1, 0)
    result_labels = data.y[[node[0] for node in nodes_new]]
    result_labels = result_labels.squeeze(dim=-1)

    data_new = Data(x=result_features, edge_index=result_edges, y=result_labels)
    node_years_new = torch.tensor([node_years[node[0]] for node in nodes_new])
    data_new.train_mask = torch.logical_and(node_years_new >= year_bound[0], node_years_new < year_bound[1])
    data_new.val_mask = torch.logical_and(node_years_new >= year_bound[1], node_years_new < year_bound[2])
    data_new.node_years = node_years_new

    return data_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare partitioned ogbn-arxiv dataset")
    parser.add_argument("--data_dir", type=str, help="which method for label shift adaptation")
    args = parser.parse_args()

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=args.data_dir)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=len(data.y))

    ls_subdir = os.path.join(args.data_dir, "ogbn_arxiv", "label_shift")
    os.makedirs(ls_subdir, exist_ok=True)
    print("Begin partition source data")
    data_src = temp_partition_arxiv(data, [0, 2014, 2015])  # src train: ~ 2011, src val: 2012
    torch.save(data_src, os.path.join(ls_subdir, "data_src.pt"))
    print("Begin partition target data")
    data_tgt = temp_partition_arxiv(data, [2018, 2019, 2020])  # tgt train: 2013 ~ 2018, tgt val: 2019
    torch.save(data_tgt, os.path.join(ls_subdir, "data_tgt.pt"))
    print("Begin partition target test data")
    test_data_tgt = temp_partition_arxiv(data, [0, 2020, 2021])  # tgt test: 2020
    test_data_tgt.test_mask = test_data_tgt.val_mask
    torch.save(test_data_tgt, os.path.join(ls_subdir, "test_data_tgt.pt"))


