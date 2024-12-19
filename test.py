from sklearn.cluster import AgglomerativeClustering,KMeans,OPTICS,SpectralClustering,DBSCAN
from matplotlib import pyplot
import numpy as np
import umap
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv

from pygod.models import DOMINANT
from torch_geometric.data import Data
import argparse
import os
#from keras import optimizers, losses
#from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import random
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier as DTC
import matplotlib.pyplot as pp
from pygod.models import CoLA
from pygod.models import DOMINANT
from torch_geometric.data import Data
from pygod.models import ANOMALOUS
from pygod.models import MLPAE
import torch_geometric.utils as Utils
from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import top_k_accuracy_score
import networkx
import torch
import torch.nn as nn
import torch.nn.functional as Fu
import torch.optim as optim
from torch.nn.parameter import Parameter
from graphxai.utils.explanation import Explanation
from graphxai.utils.nx_conversion import khop_subgraph_nx
from torch_geometric.utils import k_hop_subgraph
from sklearn.cluster import KMeans
from graphxai.datasets.utils.feature_generators import gaussian_lv_generator
import time
import math
import random
from sklearn.cluster import AgglomerativeClustering,KMeans,OPTICS,SpectralClustering,DBSCAN
from matplotlib import pyplot
import numpy as np
# train a dominant detector
from pygod.models import DOMINANT
from torch_geometric.data import Data
from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import gc

from torch_geometric.utils import to_dense_adj
from SuspiciousGCN import Both as SGCN
from SuspiciousGSage import Both as SSage
from SuspiciousGSage import Both as SSGC
from SuspiciousGAT import Both as SGAT
from ExGAD import COBAGAD
from pygod.utils.utility import validate_device
from pygod.models import AnomalyDAE
import csv
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid 
from torch_geometric.datasets import Amazon 
from torch_geometric.data import Data 
import random
from pygod.models import DOMINANT
from pygod.metrics import eval_roc_auc
import csv
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid 
from torch_geometric.datasets import Amazon 
import random
from pygod.models import DOMINANT

from pygod.metrics import eval_roc_auc

from scipy.sparse import data
from torch_geometric.data import Data
import pandas as pd
from utils import  expand_arrays, load_data_anomalies
#from models import (graph_model)
from graphxai.explainers import PGExplainer,GNNExplainer, IntegratedGradExplainer, GradExplainer, GuidedBP,RandomExplainer

def clustering(array,index_anomaly,index_normal):
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(array)
    #print(embedding)
    model =DBSCAN(eps=0.5,min_samples=2)
    # fit model and predict clusters
    #yhat = model.fit_predict(array)
    yhat = model.fit_predict(embedding)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    # create scatter plot for samples from each cluster

    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        #print(row_ix)
        # create scatter of these samples
        pyplot.scatter(embedding[row_ix, 0], embedding[row_ix, 1],alpha=0.5)
    pyplot.show()
    pyplot.scatter(embedding[index_anomaly, 0], embedding[index_anomaly, 1],c="green")
    pyplot.scatter(embedding[index_normal, 0], embedding[index_normal, 1],c="red",alpha=0.05)
    pyplot.show()
def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def count(gt,exp): 
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(0,len(exp)):
        if exp[i]==1:
            if gt[i]==exp[i]:
                TP+=1
            else:
                FP+=1
        else:
            if gt[i]==exp[i]:
                TN+=1
            else:
                FN+=1
    
    return(TP / (TP + FP + FN + 1e-09))
def countedge(gt,exp): 
    TP=0
    FP=0
    FN=0
    TN=0
    for i in exp:
        t=0
        for j in gt :
            if i==j:
                TP+=1
                t=1
        if t==0:
            FP+=1
    for j in gt:
        t=0
        for j in exp :
            if i==j:
                t=1
        if t==0:
            FN+=1
 
                
    return(TP / (TP + FP + FN + 1e-09))
def countacc(gt,exp): 
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(0,len(exp)):
        if exp[i]==1:
            if gt[i]==exp[i]:
                TP+=1
            else:
                FP+=1
        else:
            if gt[i]==exp[i]:
                TN+=1
            else:
                FN+=1
                
    return((TP) / (TP + FP  + 1e-09))
def countedgeacc(gt,exp): 
    TP=0
    FP=0
    FN=0
    TN=0
    for i in exp:
        t=0
        for j in gt :
            if i==j:
                TP+=1
                t=1
        if t==0:
            FP+=1
    for j in gt:
        t=0
        for j in exp :
            if i==j:
                t=1
        if t==0:
            FN+=1
 
    return((TP) / (TP + FP  + 1e-09))
                

def countrec(gt,exp): 
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(0,len(exp)):
        if exp[i]==1:
            if gt[i]==exp[i]:
                TP+=1
            else:
                FP+=1
        else:
            if gt[i]==exp[i]:
                TN+=1
            else:
                FN+=1
                
    return((TP) / (TP + FN + 1e-09))

def countedgerec(gt,exp): 
    TP=0
    FP=0
    FN=0
    TN=0
    for i in exp:
        t=0
        for j in gt :
            if i==j:
                TP+=1
                t=1
        if t==0:
            FP+=1
    for j in gt:
        t=0
        for j in exp :
            if i==j:
                t=1
        if t==0:
            FN+=1
 

    return((TP) / (TP + FN + 1e-09))
                
def meancountex(feat_imp,edge_imp,expl,index_anomaly,data,qs,qx):
    index_anomaly=np.where(index_anomaly==1)[0]
    moyfeature=np.empty(0)
    moyedge=np.empty(0)
    
    accfeature=np.empty(0)
    recfeature=np.empty(0)
    accedge=np.empty(0)
    recedge=np.empty(0)
    
    aucfeature=np.empty(0)
    aucedge=np.empty(0)
    
    #index_anomaly=torch.from_numpy(index_anomaly)
    for node_idx in torch.from_numpy(index_anomaly):
        #subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(node_idx, 3, data.edge_index, relabel_nodes=True)
        print(data.edge_index)
        exp = expl.get_explanation_node(node_idx = int(node_idx),num_hops=1, x = data.x, edge_index = data.edge_index)
        #gnnex_exp = gnnex.get_explanation_node(node_idx = 7,num_hops=3, x = data.x, edge_index = data.edge_index)
        moyfeature=np.append(moyfeature,count(feat_imp[node_idx],exp.feature_imp>exp.feature_imp.quantile(qx).item()))
        print(node_idx)
        #print(len(exp.edge_imp))
        #print("countedge"+str(exp.edge_reference))
        if(len(exp.edge_imp)>0):
            subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(node_idx, 1, data.edge_index)
            subSusps=np.empty(0)
            sub_edge_index=sub_edge_index.detach().cpu().numpy()
            print(sub_edge_index)
            print(edge_imp[node_idx])
        #moyedge=np.append(moyedge,countedge(edge_imp[node_idx],np.where((s[node_idx]>s[node_idx].quantile(qs)).detach().cpu().numpy())[0]))
        
        accfeature=np.append(accfeature,countacc(feat_imp[node_idx],exp.feature_imp>exp.feature_imp.quantile(qx).item()))
        #accedge=np.append(accedge,countedgeacc(edge_imp[node_idx],np.where((s[node_idx]>s[node_idx].quantile(qs)).detach().cpu().numpy())[0]))
        
        recfeature=np.append(recfeature,countrec(feat_imp[node_idx],exp.feature_imp>exp.feature_imp.quantile(qx).item()))
        #recedge=np.append(recedge,countedgerec(edge_imp[node_idx],np.where((s[node_idx]>s[node_idx].quantile(qs)).detach().cpu().numpy())[0]))
        
        #aucfeature=np.append(aucfeature,eval_roc_auc(gt[node_idx][0].feat_imp,(x[node_idx]).detach().cpu().numpy()))
        #aucedge=np.append(aucedge,eval_roc_auc(gt[node_idx][0].edge_imp,np.where(s[node_idx].detach().cpu().numpy()))
    return moyfeature.mean(),moyedge.mean(),aucfeature.mean(),aucedge.mean(),accfeature.mean(),accedge.mean(),recfeature.mean(),recedge.mean()
def meancountkex(feat_imp,edge_imp,expl,data,index_anomaly,dec):
    index_anomaly=np.where(index_anomaly==1)[0]
    moyfeat=np.empty(0)
    accfeat=np.empty(0)
    recfeat=np.empty(0)
    aucfeat=np.empty(0)
    exp=np.empty(0)
    for node_idx in index_anomaly:
        if dec:
            y=torch.from_numpy(data.y)
            y=y.long()
            exp = np.append(exp,expl.get_explanation_node(node_idx = int(node_idx),num_hops=1, x = data.x, edge_index = data.edge_index,y=y))
            subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(int(node_idx), 1, data.edge_index)            
        else:
            exp = np.append(exp,expl.get_explanation_node(node_idx = int(node_idx),num_hops=1, x = data.x, edge_index = data.edge_index))
    for q in range(1,6):
        moyfeature=np.empty(0)
        accfeature=np.empty(0)
        recfeature=np.empty(0)
        idx=0
        for node_idx in index_anomaly:
            indices = np.argpartition(exp[idx].feature_imp.detach().cpu().numpy(), -q)[-q:]
            if(idx==0):
                print(exp[idx].feature_imp)
            moyfeature=np.append(moyfeature,count(feat_imp[node_idx],exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]]))
            accfeature=np.append(accfeature,countacc(feat_imp[node_idx],exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]]))
            recfeature=np.append(recfeature,countrec(feat_imp[node_idx],exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]]))
                        
            idx+=1
        moyfeat=np.append(moyfeat,moyfeature.mean())
        accfeat=np.append(accfeat,accfeature.mean())
        recfeat=np.append(recfeat,recfeature.mean())   
    
    moyed=np.empty(0)
    acced=np.empty(0)
    reced=np.empty(0)
    auced=np.empty(0)
    for q in range(1,len(edge_imp[node_idx])):
        moyedge=np.empty(0)
        accedge=np.empty(0)
        recedge=np.empty(0)
        idx=0
        for node_idx in index_anomaly:
            save=q
            if (len(ed)>=q):
                indices = np.argpartition(exp[idx].feature_imp.detach().cpu().numpy(), -q)[-q:]
                subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(int(node_idx), 1, data.edge_index)
                moyedge=np.append(moyedge,countedge(edge_imp[node_idx],subset[indices]))
                accedge=np.append(accedge,countedgeacc(edge_imp[node_idx],subset[indices]))
                recedge=np.append(recedge,countedgerec(edge_imp[node_idx],subset[indices]))

            moyed=np.append(moyed,moyedge.mean())
            acced=np.append(acced,accedge.mean())
            reced=np.append(reced,recedge.mean())
            idx+=1
    return moyfeat,accfeat,recfeat,moyed,acced,reced

def meancount(feat_imp,edge_imp,s,x,index_anomaly,qs,qx):
    index_anomaly=np.where(index_anomaly==1)[0]
    moyfeature=np.empty(0)
    moyedge=np.empty(0)
    
    accfeature=np.empty(0)
    recfeature=np.empty(0)
    accedge=np.empty(0)
    recedge=np.empty(0)
    
    aucfeature=np.empty(0)
    aucedge=np.empty(0)
    for node_idx in index_anomaly:
        
        moyfeature=np.append(moyfeature,count(feat_imp[node_idx],x[node_idx]>=x[node_idx].quantile(qx).item()))
        moyedge=np.append(moyedge,countedge(edge_imp[node_idx],np.where((s[node_idx]>=s[node_idx].quantile(qs)).detach().cpu().numpy())[0]))
        
        accfeature=np.append(accfeature,countacc(feat_imp[node_idx],x[node_idx]>=x[node_idx].quantile(qx).item()))
        accedge=np.append(accedge,countedgeacc(edge_imp[node_idx],np.where((s[node_idx]>=s[node_idx].quantile(qs)).detach().cpu().numpy())[0]))
        
        recfeature=np.append(recfeature,countrec(feat_imp[node_idx],x[node_idx]>=x[node_idx].quantile(qx).item()))
        recedge=np.append(recedge,countedgerec(edge_imp[node_idx],np.where((s[node_idx]>=s[node_idx].quantile(qs)).detach().cpu().numpy())[0]))
        
        #aucfeature=np.append(aucfeature,eval_roc_auc(gt[node_idx][0].feat_imp,(x[node_idx]).detach().cpu().numpy()))
        #aucedge=np.append(aucedge,eval_roc_auc(gt[node_idx][0].edge_imp,np.where(s[node_idx].detach().cpu().numpy()))
    return moyfeature.mean(),moyedge.mean(),aucfeature.mean(),aucedge.mean(),accfeature.mean(),accedge.mean(),recfeature.mean(),recedge.mean()
def meancountk(feat_imp,edge_imp,s,x,index_anomaly):
    index_anomaly=np.where(index_anomaly==1)[0]
    moyfeat=np.empty(0)
    accfeat=np.empty(0)
    recfeat=np.empty(0)
    aucfeat=np.empty(0)
    for q in range(1,6):

        moyfeature=np.empty(0)
        accfeature=np.empty(0)
        recfeature=np.empty(0)
        for node_idx in index_anomaly:
            indices = np.argpartition(x[node_idx].detach().cpu().numpy(), -q)[-q:]
            moyfeature=np.append(moyfeature,count(feat_imp[node_idx],x[node_idx]>x[node_idx][indices[0]]))
            accfeature=np.append(accfeature,countacc(feat_imp[node_idx],x[node_idx]>x[node_idx][indices[0]]))
            recfeature=np.append(recfeature,countrec(feat_imp[node_idx],x[node_idx]>x[node_idx][indices[0]]))
        moyfeat=np.append(moyfeat,moyfeature.mean())
        accfeat=np.append(accfeat,accfeature.mean())
        recfeat=np.append(recfeat,recfeature.mean())
        
    moyed=np.empty(0)
    acced=np.empty(0)
    reced=np.empty(0)
    auced=np.empty(0)
    for q in range(1,6):
        moyedge=np.empty(0)
        accedge=np.empty(0)
        recedge=np.empty(0)
        for node_idx in index_anomaly:
            
            indices = np.argpartition(s[node_idx].detach().cpu().numpy(), -q)[-q:]
            
            moyedge=np.append(moyedge,countedge(edge_imp[node_idx],indices))
            accedge=np.append(accedge,countedgeacc(edge_imp[node_idx],indices))
            recedge=np.append(recedge,countedgerec(edge_imp[node_idx],indices))
                
        moyed=np.append(moyed,moyedge.mean())
        acced=np.append(acced,accedge.mean())
        reced=np.append(reced,recedge.mean())
        
    return moyfeat,accfeat,recfeat,moyed,acced,reced



# uncomment to draw decision trees
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus


class MyGNN(torch.nn.Module):
    def __init__(self,input_feat, hidden_channels, classes = 2):
        super(MyGNN, self).__init__()
        #self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        #self.gin1 = GINConv(self.mlp_gin1)
        self.gat1 = GATConv(in_channels=input_feat,out_channels=hidden_channels)
        #self.mlp_gin2 = torch.nn.Linear(hidden_channels, classes)
        #self.gin2 = GINConv(self.mlp_gin2)
        self.gat2= GATConv(in_channels=hidden_channels,out_channels=classes)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gat1(x, edge_index)
        x = x.relu()
        x = self.gat2(x, edge_index)
        return x
def trainspec(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, data: Data,idx_train, losses: list = None):
    model.train()
    idx_train=torch.from_numpy(idx_train)
    y=torch.from_numpy(data.y)
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # print('Out shape', out.shape)
    # print('y shape', data.y.shape)
    y=y.long()
    loss = criterion(out[idx_train], y[idx_train])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    if losses is not None:
        losses.append(loss.item())

    return loss
def testspec(model: torch.nn.Module, data: Data,idx_test, test_accs: list = None):
    test_accs=[]
    y=torch.from_numpy(data.y)
    
    y=y.long()
    idx_test=torch.from_numpy(idx_test)
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    print(idx_test)
    s=0
    t=0
    for i in range(0,len(idx_test)):
        if (idx_test[i] ):  # Check against ground-truth labels.
            t+=1    
            if ((pred[i] == y[i])):  # Check against ground-truth labels.
                s+=1
    test_acc = s / t # Derive ratio of correct predictions.
    test_accs.append(test_acc)
    
    return pred


def to_k_highest(score,k):
    score=score.flatten()
    score=score.tolist()
    score.sort(reverse=True)
    threshold=score[k-1]
    score=[1 if x >= threshold else 0 for x in score]
    return score

def X_modif(G, X_initial, id, reroll_pourcentage, blank_vector):
    X_bis = np.copy(X_initial)
    neighbors_of_id = list(G.neighbors(id))
    neighbors_of_id.append(id)
    for j in neighbors_of_id:
        if random.random() < reroll_pourcentage:
            X_bis[j] = blank_vector
    return X_bis


    sizeimp=0
    # choose dataset to load
    # A: adjacency matrix	N x N
    # X: feature matrix	  N x F
    # labels: one-hot labels for every node	  N x n_comm
    # Y_train: labels but vectors are zeros if nodes not in training set   N x n_comm
    # idx_train: vector, 1 if node i is in training set, 0 otherwise   N

    N = X.shape[0]	# number of nodes
    F = X.shape[1]	# number of features
    y=np.zeros((X.shape[0]))
    for i in anomalies_list:
        y[i]=1
    for i in normal_list:
        y[i]=0
    n_comm = len(Y_train[0])  # number of classes
    batch_size = N	# set batch_size to number of nodes



X=pd.read_csv("X1.csv",sep=';',header=0,low_memory=False)
X=X.replace("FALSE",False)
X=X.replace("TRUE",True)
print(X.shape[0])
ei=pd.read_csv("EdgesACT.csv",sep=',',index_col=0)
print(ei)
y=np.zeros(X.shape[0])
gt=pd.read_csv("R09.csv",sep=",",index_col=0)
y[gt]=True
print(y.sum()/X.shape[0])

edge_index=torch.LongTensor(ei.values.T)
x=torch.FloatTensor(X.astype(float).values)
        
data=Data(x=x,edge_index=edge_index,y=y)

index_normal_train = np.empty(0)
index_anomaly_train = np.empty(0)
        
for i in range(0,round(X.shape[0]/10)):
    if y[i]:
        index_anomaly_train=np.append(index_anomaly_train,i)
    else :
        index_normal_train=np.append(index_normal_train,i)
sample_s =torch.LongTensor(index_anomaly_train)
sample_n =torch.LongTensor(index_normal_train)
print("i survived 1")
model = SGAT(Sample_n=sample_n, Sample_s= sample_s,epoch=50,hid_dim=5,gpu=0,dropout=0.5,alpha=0.5,batch_size=0)
model.fit(data)
Rn,xh,sh,at,ad = model.decision_function(data)
Rn=np.nan_to_num(Rn)
print("i survived 2")
auc_score = eval_roc_auc(data.y, Rn)
print(auc_score)
Smoyfeat,Saccfeat,Srecfeat,Smoyed,Sacced,Sreced=meancountk(feat_imp,edge_imp,sh,xh,yn)

model = SGAT(Sample_n=sample_s, Sample_s= sample_n,epoch=50,hid_dim=5,gpu=0,dropout=0.5,alpha=0.5,batch_size=0)
model.fit(data)
print("i survived 3")
Rs,xh,sh,at,ad = model.decision_function(data)
auc_score = eval_roc_auc(data.y, Rs)
print(auc_score)       