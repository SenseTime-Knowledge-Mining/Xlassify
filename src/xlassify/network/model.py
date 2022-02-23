__version__ = '1.0.0'
__all__ = ['ResidualModel', 'RNA_Model', 'CNN_Model','MLP_Model']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList


class ResidualModel(nn.Module):
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        super(ResidualModel, self).__init__()
        '''DNN'''
        DNN_n_layers = 4
        self.DNN_h_dim = [self.ft_dim, self.h_dim, self.h_dim, self.h_dim, self.n_label]
        self.DNN_layer_lst = []
        for idx in range(DNN_n_layers):
            hidden_nn_lst = []
            hidden_nn_lst.append(nn.Linear(self.DNN_h_dim[idx], self.DNN_h_dim[idx+1]))
            if idx != DNN_n_layers - 1:
                hidden_nn_lst.append(nn.ELU())
                hidden_nn_lst.append(nn.Dropout(p=self.dropout))
            self.hidden_nn = nn.Sequential(*hidden_nn_lst)
            self.DNN_layer_lst.append(self.hidden_nn)
        self.dnn = ModuleList(self.DNN_layer_lst)
        
    def forward(self, data):
        data0 = self.dnn[0](data)
        data1 = self.dnn[1](data0)
        data2 = self.dnn[2](data1) 
        data = data0 + data2
        data = self.dnn[3](data) 
        return data

class CNN_Model(nn.Module):
    def __init__(self, feat_size, filters, kernels, dropout):
        super(CNN_Model, self).__init__()
        self.conv_1 = nn.Sequential(
        nn.Conv1d(in_channels = feat_size, out_channels = filters[0], kernel_size = kernels[0], padding = kernels[0]//2), # 256, 256, 1748
        nn.BatchNorm1d(filters[0]),
        nn.ReLU(),
        nn.Dropout(dropout),
        )
        self.conv_2 = nn.Sequential(
        nn.Conv1d(in_channels = filters[0], out_channels = filters[1], kernel_size = kernels[1], padding = kernels[1]//2),
        nn.BatchNorm1d(filters[1]),
        nn.ReLU(),
        nn.Dropout(dropout),
    )
    def forward(self, seq_mat): # batch_size * max_seq_len * feat_size
        seq_mat = seq_mat.transpose(1, 2)
        seq_mat = self.conv_1(seq_mat)
        seq_mat = self.conv_2(seq_mat)  
        res, _ = seq_mat.max(dim=-1) # max_pooling?
        return res
        
class MLP_Model(nn.Module): 
    def __init__(self, feat_size, n_targets, hidden_size, dropout):
        super(MLP_Model, self).__init__()
        self.dense_1 = nn.Sequential(
            nn.BatchNorm1d(feat_size),
            nn.Linear(feat_size, hidden_size),
            nn.ELU(),
        )
        self.dense_2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.dense_3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_targets),
            nn.ELU(),
        )     
    def forward(self, seq_mer):
        seq_mer = self.dense_1(seq_mer)
        seq_mer = self.dense_2(seq_mer)
        res = self.dense_3(seq_mer)
        return res

class RNA_Model(nn.Module):
    def __init__(self, use_model, seq_mat_dim, seq_mer_dim, n_targets, hidden_size, dropout):
        super(RNA_Model, self).__init__()
        filters = [256, 64]
        kernels = [7, 15]
        cnn_input_dim = seq_mat_dim
        cnn_output_dim = filters[-1]
        mlp_input_dim = seq_mer_dim
        mlp_output_dim = filters[-1] 
        # mlp_output_dim = n_targets
        fc_input_dim = 256
        self.use_model = use_model
        self.cnn = CNN_Model(cnn_input_dim, filters, kernels, dropout)
        self.mlp = MLP_Model(mlp_input_dim, mlp_output_dim, hidden_size, dropout)

        if self.use_model == 'CNN':
            self.linear_1 = nn.Linear(cnn_output_dim, fc_input_dim)
        elif self.use_model == 'MLP':
            self.linear_1 = nn.Linear(mlp_output_dim, fc_input_dim)
        else:
            # self.linear_1 = nn.Linear(mlp_output_dim + cnn_output_dim, fc_input_dim) # cat
            self.linear_1 = nn.Linear(cnn_output_dim, fc_input_dim) # other fusion
        self.linear_2 = nn.Linear(fc_input_dim, n_targets)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, seq_mat, seq_mer):
        h_mat = self.cnn(seq_mat)
        h_mer = self.mlp(seq_mer)
        
        if self.use_model == 'CNN':
            res = h_mat
        elif self.use_model == 'MLP':
            res = h_mer
        else:
            # flag = torch.cat((h_mat, h_mer), 1)
            res = (h_mat+h_mer) + h_mat*h_mer

        res = self.linear_1(res)
        res = self.activation(res)
        res = self.dropout(res)

        res = self.linear_2(res)
        return res