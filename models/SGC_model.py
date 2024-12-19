import torch.nn as nn

from torch_geometric.nn.conv import SSGConv
class SGC(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, graph) -> None:
        super().__init__()
        self.nn1 = nn.Linear(in_dim, hid_dim)
        self.gin1 = SSGConv(self.nn1)
        self.nn2 = nn.Linear(hid_dim, out_dim)
        self.gin2 = SSGConv(self.nn1)
        self.edge_index = graph.edge_index

    def forward(self, x):
        hid = self.gin1(x, self.edge_index)
        x = self.gin2(hid, self.edge_index)
        return x, hid
