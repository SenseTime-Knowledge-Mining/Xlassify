import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
# from models.tabnet import *
class Model(nn.Module):
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        super(Model, self).__init__()
        '''DNN'''
        DNN_n_layers = 3
        self.DNN_h_dim = [self.ft_dim, self.h_dim, self.h_dim, self.n_label]
        self.DNN_layer_lst = []
        for idx in range(DNN_n_layers):
            self.DNN_layer_lst.append(nn.Linear(self.DNN_h_dim[idx], self.DNN_h_dim[idx+1]))
            if idx != DNN_n_layers - 1:
                # self.DNN_layer_lst.append(nn.BatchNorm1d(self.DNN_h_dim[idx+1]))
                self.DNN_layer_lst.append(nn.ELU())
                self.DNN_layer_lst.append(nn.Dropout(p=self.dropout))
        self.dnn = nn.Sequential(*self.DNN_layer_lst)

        '''Transformer'''
        # n_layers = 1 # temp
        # ntoken, ninp = 4, 1
        # self.encoder = nn.Embedding(ntoken, ninp)
        # encoder_layer = nn.TransformerEncoderLayer(ninp, 1, self.h_dim, self.dropout, 'relu') # relu or gelu
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # self.decoder = nn.Linear(ninp, ntoken)
        # self.fc = nn.Linear(self.ft_dim, self.n_label)

    def forward(self, data):
        data = self.dnn(data)
        # data = data.unsqueeze(2)
        # data = self.transformer_encoder(data)
        # data = self.fc(data.view(batch_size, -1))
        return data

class MLPModel(nn.Module):
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        super(MLPModel, self).__init__()
        '''DNN'''
        DNN_n_layers = 3
        self.DNN_h_dim = [self.ft_dim, self.h_dim, self.h_dim, self.n_label]
        self.DNN_layer_lst = []
        for idx in range(DNN_n_layers):
            self.DNN_layer_lst.append(nn.Linear(self.DNN_h_dim[idx], self.DNN_h_dim[idx+1]))
            if idx != DNN_n_layers - 1:
                self.DNN_layer_lst.append(nn.BatchNorm1d(self.DNN_h_dim[idx+1]))
                self.DNN_layer_lst.append(nn.ELU())
                self.DNN_layer_lst.append(nn.Dropout(p=self.dropout))
        self.dnn = nn.Sequential(*self.DNN_layer_lst)

    def forward(self, data):
        return self.dnn(data)

class CNNModel(nn.Module):
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        super(CNNModel, self).__init__()

        '''CNN'''
        # self.act = nn.Sigmoid() # S
        # self.act = nn.ReLU() # R
        self.act = nn.ELU() # E
        self.filters = [1,32,64]
        self.kernel_size = [15, 3, 49, 3]
        self.seq_cnn = nn.Sequential(
            (nn.Conv1d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=self.kernel_size[0], padding = self.kernel_size[0]//2)  ),
            (nn.BatchNorm1d(self.filters[1])),
            (self.act),
            (nn.MaxPool1d(kernel_size=self.kernel_size[1], stride=self.kernel_size[1])),
            (nn.Conv1d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=self.kernel_size[2], padding = self.kernel_size[2]//2)  ),
            (nn.BatchNorm1d(self.filters[2])),
            (self.act),
            (nn.MaxPool1d(kernel_size=self.kernel_size[3], stride=self.kernel_size[3])),
        )
        # dim = self.ft_dim #out = (n, out_channels, [(d-k)/stride+1])
        # for i,k in enumerate(self.kernel_size):
        #     if i%2==0:
        #         dim-=k-1
        #     else:
        #         if k != 0:
        #             dim//=k
        dim = 1820
        '''DNN'''
        self.DNN_layer_num = 2
        self.DNN_h_dim = [self.filters[-1]*dim, 256, self.n_label]
        self.DNN_layer_lst = []
        for idx in range(self.DNN_layer_num):
            self.DNN_layer_lst.append(nn.Linear(self.DNN_h_dim[idx], self.DNN_h_dim[idx+1]))
            if idx != self.DNN_layer_num - 1:
                self.DNN_layer_lst.append(nn.BatchNorm1d(self.DNN_h_dim[idx+1]))
                self.DNN_layer_lst.append(nn.ELU())
                self.DNN_layer_lst.append(nn.Dropout(p=self.dropout))
            # else:
                # self.DNN_layer_lst.append(nn.LogSoftmax(dim=1))
        self.seq_dnn = nn.Sequential(*self.DNN_layer_lst)
        
    def forward(self, RNA_ft):
        batch_size = RNA_ft.shape[0]
        device = RNA_ft.device
        RNA_ft = RNA_ft.unsqueeze(1)
        RNA_ft = self.seq_cnn(RNA_ft).view(batch_size, -1)
        # print(RNA_ft.shape[1]/self.filters[-1])
        RNA_ft = self.seq_dnn(RNA_ft)
        return RNA_ft
# cnn = CNNModel(4**7, datasetG.n_label)
# data1 = torch.tensor(datasetG.kmer_mat)[:4]
# print(data1.shape)
# cnn(data1)



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
                hidden_nn_lst.append(nn.BatchNorm1d(self.DNN_h_dim[idx+1]))
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

# class TabNetModel(nn.Module):
#     def __init__(self, **kwargs):
#         for (key, value) in kwargs.items():
#             setattr(self, key, value)
#         super(TabNetModel, self).__init__()

#         '''TabNet'''
#         self.tabnet = TabNet(
#             input_dim=self.ft_dim,
#             output_dim=self.n_label,
#             n_independent=2,
#             n_shared=0,
#             hidden_dim=256,
#             dropout=0.5,
#             n_steps=2,
#             momentum=0.1
#             )

#     def forward(self, data):
#         data = self.tabnet(data)
#         return data