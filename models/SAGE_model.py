import torch.nn as nn

from torch_geometric.nn.conv import SAGEConv
class Sage(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, graph) -> None:
        super().__init__()
        self.nn1 = nn.Linear(in_dim, hid_dim)
        self.gin1 = SAGEConv(in_channels=in_dim,out_channels=hid_dim)
        self.nn2 = nn.Linear(hid_dim, out_dim)
        self.gin2 = SAGEConv(in_channels=hid_dim,out_channels=out_dim) 
        self.edge_index = graph.edge_index

    def forward(self, x):
        hid = self.gin1(x, self.edge_index)
        x = self.gin2(hid, self.edge_index)
        return x, hid
