from sklearn.cluster import AgglomerativeClustering,KMeans,OPTICS,SpectralClustering,DBSCAN
from matplotlib import pyplot
import numpy as np
import umap
import torch
import matplotlib.pyplot as plt
from graphxai.datasets import ShapeGGen
from torch_geometric.nn import GINConv
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

from SuspiciousGCN import Both as SGCN
from SuspiciousGSage import Both as SSage
from SuspiciousGSage import Both as SSGC
from SuspiciousGAT import Both as SGAT
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
    exp=exp.detach().cpu().numpy()
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
    print(TP)
    return(TP / (TP + FP + FN + 1e-09))
def countedge(gt,exp): 
    TP=0
    FP=0
    FN=0
    TN=0
    exp=exp.detach().cpu().numpy()
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
    exp=exp.detach().cpu().numpy()
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
    exp=exp.detach().cpu().numpy()
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
    exp=exp.detach().cpu().numpy()
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
    exp=exp.detach().cpu().numpy()
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
        subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(node_idx, 3, data.edge_index, relabel_nodes=True)
            
        exp = expl.get_explanation_node(node_idx = int(node_idx),num_hops=3, x = data.x, edge_index = data.edge_index)
        #gnnex_exp = gnnex.get_explanation_node(node_idx = 7,num_hops=3, x = data.x, edge_index = data.edge_index)
        moyfeature=np.append(moyfeature,count(feat_imp[node_idx],exp.feature_imp>=exp.feature_imp.quantile(qx).item()))
        #print(exp.feature_imp)
        if(len(edge_imp[node_idx])>0):
            subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(node_idx, 3, data.edge_index, relabel_nodes=True)
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
            exp = np.append(exp,expl.get_explanation_node(node_idx = int(node_idx),num_hops=3, x = data.x, edge_index = data.edge_index,y=y))
        else:
            exp = np.append(exp,expl.get_explanation_node(node_idx = int(node_idx),num_hops=3, x = data.x, edge_index = data.edge_index))
    for q in range(0,6):
        moyfeature=np.empty(0)
        accfeature=np.empty(0)
        recfeature=np.empty(0)
        idx=0
        for node_idx in index_anomaly:
            indices = np.argpartition(exp[idx].feature_imp.detach().cpu().numpy(), -q)[-q:]
            moyfeature=np.append(moyfeature,count(feat_imp[node_idx],exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]])*q/(exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]]).sum())
            accfeature=np.append(accfeature,countacc(feat_imp[node_idx],exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]])*q/(exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]]).sum())
            recfeature=np.append(recfeature,countrec(feat_imp[node_idx],exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]])*q/(exp[idx].feature_imp>=exp[idx].feature_imp[indices[0]]).sum())
            idx+=1
        moyfeat=np.append(moyfeat,moyfeature.mean())
        accfeat=np.append(accfeat,accfeature.mean())
        recfeat=np.append(recfeat,recfeature.mean())
        
    moyed=np.empty(0)
    acced=np.empty(0)
    reced=np.empty(0)
    auced=np.empty(0)
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
    for q in range(0,6):

        moyfeature=np.empty(0)
        accfeature=np.empty(0)
        recfeature=np.empty(0)
        for node_idx in index_anomaly:
            indices = np.argpartition(x[node_idx].detach().cpu().numpy(), -q)[-q:]
            moyfeature=np.append(moyfeature,count(feat_imp[node_idx],x[node_idx]>=x[node_idx][indices[0]])*q/(x[node_idx]>=x[node_idx][indices[0]]).int().detach().cpu().numpy().sum())
            accfeature=np.append(accfeature,countacc(feat_imp[node_idx],x[node_idx]>=x[node_idx][indices[0]])*q/(x[node_idx]>=x[node_idx][indices[0]]).int().detach().cpu().numpy().sum())
            recfeature=np.append(recfeature,countrec(feat_imp[node_idx],x[node_idx]>=x[node_idx][indices[0]])*q/(x[node_idx]>=x[node_idx][indices[0]]).int().detach().cpu().numpy().sum())
        moyfeat=np.append(moyfeat,moyfeature.mean())
        accfeat=np.append(accfeat,accfeature.mean())
        recfeat=np.append(recfeat,recfeature.mean())
        
    moyed=np.empty(0)
    acced=np.empty(0)
    reced=np.empty(0)
    auced=np.empty(0)
    for q in range(0,6):
        moyedge=np.empty(0)
        accedge=np.empty(0)
        recedge=np.empty(0)
        for node_idx in index_anomaly:
            
            indices = np.argpartition(s[node_idx].detach().cpu().numpy(), -q)[-q:]
            
            #moyedge=np.append(moyedge,countedge(edge_imp[node_idx],indices))
            #accedge=np.append(accedge,countedgeacc(edge_imp[node_idx],indices))
            #recedge=np.append(recedge,countedgerec(edge_imp[node_idx],indices))
                
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
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, classes)
        self.gin2 = GINConv(self.mlp_gin2)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
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


def sample_mask(idx, l):
	"""Create mask."""
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=bool)

def split(G):
    number_of_attributes = len(G.x[0])
    number_of_nodes = len(G.y)
    number_of_communities = 2  # anomaly and non-anomaly
    adj = nx.to_numpy_array(G.G)
    features = np.zeros((number_of_nodes, number_of_attributes))
    normal_list = []
    anomalies_list = []
    for i in range(len(G.y)):
      if not G.y[i]:	 # the other nodes are normal
        normal_list.append(i)
      else :
        anomalies_list.append(i)
    labels = [[0, 1] for i in range(number_of_nodes)]  # labels of normal nodes
    for i in anomalies_list:
    	labels[i] = [1, 0]	# labels of anomalies
    labels = np.array(labels)
    features = G.x.detach().cpu().numpy()
    
    random.seed()  # change the seed of random to change sampling
    anomalies_list = random.sample(anomalies_list, len(anomalies_list))	 # shuffle the list of anomalies
    normal_list = random.sample(normal_list, len(normal_list))	# shuffle the list of normal nodes
    
    c = len(anomalies_list)
    d = len(normal_list)
    
    idx_test = list(anomalies_list[int(0.75 * c):]) + list(normal_list[int(0.75 * d):])
    d = c  # re-equilibrate train and validation
    idx_train = list(anomalies_list[:int(0.5 * c)]) + list(normal_list[:int(0.5 * d)])
    idx_val = list(anomalies_list[int(0.5 * c):int(0.75 * c)]) + list(normal_list[int(0.5 * d):int(0.75 * d)])
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print('Number of nodes: ')
    print(G.G.number_of_nodes())
    print('Number of edges: ')
    print(G.G.number_of_edges())
    print('Number of anomalies: ')
    print(len(anomalies_list))
    expl=[]
    feat_imp=G.explanations[0][0].feature_imp.detach().cpu().numpy()
    edge_imp=[]
    for i in range(1,G.G.number_of_nodes()):
      feat_imp=np.vstack([feat_imp,G.explanations[i][0].feature_imp.detach().cpu().numpy()])
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, anomalies_list, normal_list,expl,feat_imp, edge_imp,

def main(model_save_path: str,
         G,
         num_epochs: int,
         learning_rate: float,
         load_weights_only: bool,
         self_loops: bool,
         algo: int,
         anomaly_param: int
         ):
    sizeimp=0
    graph_name = "ShapeGGenGATS"
    F = open(G.name+"_"+str(anomaly_param) +".csv","a")
    writer=csv.writer(F,dialect='excel',delimiter=';',lineterminator = '\n')
    for k in range(0,10):
        # add self-loops in the name if necessary
        A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test, labels, anomalies_list, normal_list,expl,feat_imp,edge_imp= split(G)
        N = X.shape[0]	# number of nodes
        F = X.shape[1]	# number of features
        y=np.zeros((X.shape[0]))
        for i in anomalies_list:
            y[i]=1
        for i in normal_list:
            y[i]=0
        n_comm = len(Y_train[0])  # number of classes
        batch_size = N	# set batch_size to number of nodes
        
        Y_train, Y_val, Y_test, labels = expand_arrays(Y_train), expand_arrays(Y_val), expand_arrays(Y_test), expand_arrays(labels)	 # expand size of labels
        df = pd.DataFrame(G.G.edges())
        edge_index=torch.IntTensor(np.array(df).astype(float).T).to(torch.int64)
        x=torch.FloatTensor(X.astype(float))
        
        data=Data(x=x,edge_index=edge_index,y=y)
        data.train_mask=torch.from_numpy(Y_train)
        index_normal_train = np.empty(0)
        index_anomaly_train = np.empty(0)
        yn=y.astype(int)    
        for i in range(0,len(idx_train)):
            if idx_train[i]:
                if yn[i] == False:
                    index_normal_train=np.append(index_normal_train,i)
                else :
                    index_anomaly_train=np.append(index_anomaly_train,i)
        sample_s =torch.LongTensor(index_anomaly_train)
        sample_n =torch.LongTensor(index_normal_train)
        #return D,x,s,data.y
        #yn=np.empty(0)
        
    #     print("result Norm")
       
        model = MyGNN(data.x.shape[1], 32)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train model:
        for _ in range(100):
            loss = trainspec(model, optimizer, criterion, data,idx_train)
        
        # Final testing performance:
        pred = testspec(model, data,idx_test)
        
        auc_score = eval_roc_auc(yn[idx_test], pred[idx_test])
        print(auc_score)
        
        igex = IntegratedGradExplainer(model, criterion=criterion)
        #rint(meancountex(feat_imp,edge_imp,igex,yn,data,0.1,0.001))    
        
        igmoyfeat,igaccfeat,igrecfeat,igmoyed,igacced,igreced=meancountkex(feat_imp,edge_imp,igex,data,yn,True)
        
        gnnex = GNNExplainer(model)
        #print(meancountex(feat_imp,edge_imp,gnnex,yn,data,0.1,0.001))    
        gnnmoyfeat,gnnaccfeat,gnnrecfeat,gnnmoyed,gnnacced,gnnreced=meancountkex(feat_imp,edge_imp,gnnex,data,yn,False)
        
        #gradex = GradExplainer(model, criterion=criterion)
        #print(meancountex(feat_imp,edge_imp,gradex,yn,data,0.1,0.001))    
        #gradmoyfeat,gradaccfeat,gradrecfeat,gradmoyed,gradacced,gradreced=meancountkex(feat_imp,edge_imp,gradex,data,yn,True)
        
        randex= RandomExplainer(model)
        #print(meancountex(feat_imp,edge_imp,randex,yn,data,0.1,0.001))    
        randmoyfeat,randaccfeat,randrecfeat,randrandmoyed,randacced,randreced=meancountkex(feat_imp,edge_imp,randex,data,yn,False)
        
        
        sample_s =torch.LongTensor(index_anomaly_train)
        sample_n =torch.LongTensor(index_normal_train)
        #return D,x,s,data.y
        #yn=np.empty(0)
        
    #     print("result Norm")
        model = SGAT(Sample_n=sample_n, Sample_s= sample_s,epoch=50,hid_dim=5,gpu=0,dropout=0.5,alpha=0.5,batch_size=0)
        model.fit(data)
        Rn,xh,sh,at,ad = model.decision_function(data)
        Rn=np.nan_to_num(Rn)
        
        Smoyfeat,Saccfeat,Srecfeat,Smoyed,Sacced,Sreced=meancountk(feat_imp,edge_imp,sh,xh,yn)
        
        model = SGAT(Sample_n=sample_s, Sample_s= sample_n,epoch=50,hid_dim=5,gpu=0,dropout=0.5,alpha=0.5,batch_size=0)
        model.fit(data)
        Rs,xh,sh,at,ad = model.decision_function(data)
        
        modeld = DOMINANT(epoch=100,hid_dim=2,dropout=0.5,alpha=0.5,batch_size=0)
        modeld.fit(data)
        D,x,s=modeld.decision_function(data)
        auc_score = eval_roc_auc(data.y, D)
        print("DOM :"+str(auc_score))
        #print(D)
        #print(x)    
        Dmoyfeat,Daccfeat,Drecfeat,Dmoyed,Dacced,Dreced=meancountk(feat_imp,edge_imp,s,x,yn)
        #gea=plt.figure()
        #plt.rcParams["figure.figsize"] = [7.50, 3.50]
        #plt.rcParams["figure.autolayout"] = True
        #l1=plt.plot(range(0,6), gnnmoyfeat, label="Gnnex")
        #l2=plt.plot(range(0,6), randmoyfeat, label="Randex")
        #l5=plt.plot(range(0,6), Dmoyfeat, label="DomRe")
        #l6=plt.plot(range(0,6), Smoyfeat, label="SuspRe")
        #l3=plt.plot(range(0,6), igmoyfeat, label="Igrad")
        #l4=plt.plot(range(1,5), gradaccfeat, label="grad")
        #plt.title(G.name+" "+str(anomaly_type) +" GEA on features")
        #plt.ylim(0,1)
        #plt.xticks(range(1,6))
        #leg = plt.legend(loc='upper center')
        #plt.savefig(G.name+"_"+str(anomaly_param) +"_GEAFeat.pdf", format="pdf", bbox_inches="tight")
        #plt.close(gea)
        #pre=plt.figure()
        #plt.rcParams["figure.figsize"] = [7.50, 3.50]
        #plt.rcParams["figure.autolayout"] = True
        #l1=plt.plot(range(0,6), gnnaccfeat, label="Gnnex")
        #l2=plt.plot(range(0,6), randaccfeat, label="Randex")
        #l5=plt.plot(range(0,6), Daccfeat, label="DomRe")
        #l6=plt.plot(range(0,6), Saccfeat, label="SuspRe")
        #l3=plt.plot(range(0,6), igaccfeat, label="Igrad")
        #l4=plt.plot(range(1,5), gradaccfeat, label="grad")
        #plt.title(G.name+" "+str(anomaly_type) +" Precision on features")
        #plt.ylim(0,1)
        #plt.xticks(range(1,6))
        #leg = plt.legend(loc='upper center')
        #plt.savefig(G.name+"_"+str(anomaly_param) +"_PrecFeat.pdf", format="pdf", bbox_inches="tight")
        #plt.close(pre)
        #rec=plt.figure()
        #plt.rcParams["figure.figsize"] = [7.50, 3.50]
        #plt.rcParams["figure.autolayout"] = True
        #l1=plt.plot(range(0,6), gnnrecfeat, label="Gnnex")
        #l2=plt.plot(range(0,6), randrecfeat, label="Randex")
        #l5=plt.plot(range(0,6), Drecfeat, label="DomRe")
        #l6=plt.plot(range(0,6), Srecfeat, label="SuspRe")
        #l3=plt.plot(range(0,6), igrecfeat, label="Igrad")
        #l4=plt.plot(range(1,5), gradrecfeat, label="grad")
        #plt.title(G.name+" "+str(anomaly_type) +" Recall on features")
        #plt.ylim(0,1)
        #plt.xticks(range(1,6))
        #leg = plt.legend(loc='upper center')
        #plt.savefig(G.name+"_"+str(anomaly_param) +"_RECFeat.pdf", format="pdf", bbox_inches="tight")
        #plt.close(rec)
        write2=np.empty(0)	
        write2=np.append(write2,gnnmoyfeat)
        write2=np.append(write2,randmoyfeat)
        write2=np.append(write2,Dmoyfeat)
        write2=np.append(write2,Smoyfeat)
        write2=np.append(write2,igmoyfeat)
        write2=np.append(write2,gnnaccfeat)
        write2=np.append(write2,randaccfeat)
        write2=np.append(write2,Daccfeat)
        write2=np.append(write2,Saccfeat)
        write2=np.append(write2,igaccfeat)
    
        write2=np.append(write2,gnnrecfeat)
        write2=np.append(write2,randrecfeat)
        write2=np.append(write2,Drecfeat)
        write2=np.append(write2,Srecfeat)
        write2=np.append(write2,igrecfeat)
        writer.writerow(write2)	
        print(write2)
    
    return Rn/Rs, data.y, sample_s
graph_number = 0  # start
number_of_graphs =6	# total number of graphs
graph_number =0
number_of_times_a_graph_is_computed = 5	  # number of computations per graph
max_graph_number = number_of_graphs * number_of_times_a_graph_is_computed	 # total number of computations
G=ShapeGGen(
        base_graph= "ba",
        verify=False,
        max_tries_verification=5,
        homophily_coef=1.0,
        seed=0,
        shape_method="house",
        sens_attribution_noise=0.3,
        num_hops=3,
        model_layers=3,
        make_explanations=True,
        variant=1,
        num_subgraphs=300,
        prob_connection=0.003,
        subgraph_size=11,
        # Features=
        class_sep=15.,
        n_features=20,
        n_clusters_per_class=2,
        n_informative=2,
        add_sensitive_feature=False,
        attribute_sensitive_feature=False,
)
# relabel nodes
#nodes = list(G.nodes())
#mapping = {}
#j = 0
#for i in nodes:
#    mapping[i] = j
#    j += 1
#G = nx.relabel_nodes(G, mapping, copy=True)
D,y,s=main(model_save_path=".//",
     num_epochs=100,
     learning_rate=0.01,
     G=G,
     load_weights_only=True,
     self_loops=False,
     algo=0,
     anomaly_param=0)
c=0
yn=np.empty(0)
print("Result Suspicious")
auc_score = eval_roc_auc(y, D)
print(auc_score)


