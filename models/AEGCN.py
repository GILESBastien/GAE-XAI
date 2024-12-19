import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from pygod.models import BaseDetector

from pygod.utils.utility import validate_device
from pygod.metrics import eval_roc_auc
from models.GCN_model import GCN

class AutoEncoder(nn.Module):
	def __init__(self,
				 in_dim,
				 hid_dim,
				 num_layers,
				 dropout,
				 act):
		super(Both_Base, self).__init__()

		# split the number of layers for the encoder and decoders
		decoder_layers = int(num_layers / 2)
		encoder_layers = num_layers - decoder_layers
		self.shared_encoder = GATConv(in_channels=in_dim,
								  out_channels=hid_dim,
								
								  )

		self.attr_decoder = GATConv(in_channels=hid_dim,
								out_channels=in_dim,
								)

		self.struct_decoder = GATConv(in_channels=hid_dim,
								  out_channels=in_dim,
								  )

	def forward(self, x, edge_index):
		# encode
		h = self.shared_encoder(x, edge_index)
		# decode feature matrix
		x_ = self.attr_decoder(h, edge_index)
		# decode adjacency matrix
		h_ = self.struct_decoder(h, edge_index)
		s_ = h_ @ h_.T
		# return reconstructed matrices
		return x_, s_
