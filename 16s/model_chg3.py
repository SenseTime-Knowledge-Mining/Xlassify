# -*- coding: utf-8 -*-
import os
import sys
import pdb
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')

from kmer import kmer_featurization
from model import RNA_Model

parser = argparse.ArgumentParser(description='')
parser.add_argument('--log', type=str, help='must have logname')
parser.add_argument('--use_model', type=str, default='CNN+MLP', help='MLP,CNN, CNN+MLP')
parser.add_argument('-l', '--label_type', type=str, default='species', help='genus, species')
parser.add_argument('-k', type=int, default=7, help='kmer')
# fixed
parser.add_argument('--data_type', type=str, default='full', help='full, all, partial')
parser.add_argument('--seq_type', type=str, default='16S', help='16S, V3V4')
parser.add_argument('--threshold', type=int, default=10, help='cutoff=5, 10')
args = parser.parse_args()
k, label_type, use_model, logname = args.k, args.label_type, args.use_model, args.log
data_type, seq_type, cutoff = args.data_type, args.seq_type, args.threshold

# =========== parameter settings
global_seed = 42
kfold_seeds = [42, 2077] 
kfold_type = 'StratifiedKFold' # KFold StratifiedKFold
n_splits = cutoff # 10
# ===== train
EPOCHS = 100 # 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
LAMBDA_L2 = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
# ===== model
dropout = 0.5
hidden_size = 1024
# =========== load and save
# my_data_path = 'UHGG_16S_data'
my_data_path = 'data'
yw_data_path = 'data'
now_time = str(datetime.datetime.now()).split(" ")[0]
save_res_file = osp.join('res', logname + '_' + now_time)
save_model_path = 'model_params'
if not osp.exists(save_model_path):
    os.mkdir(save_model_path)
    
if seq_type=="16S":
    full_data = pd.read_csv( os.path.join(yw_data_path, 'full_16S_rRNA.csv'),header=None,sep="\t")
    partial_data = pd.read_csv(os.path.join(yw_data_path, 'partial_16S_rRNA.csv'),header=None,sep="\t")
    full_data.columns = ["id","type","sequence"]
    partial_data.columns = ["id","type","sequence"]
elif seq_type=="V3V4":
    v3v4_data = pd.read_csv(os.path.join(yw_data_path, '16S_rRNA_V3V4.csv'),header=None,sep="\t")
    v3v4_data.columns = ["id","type","sequence"]
    full_data = v3v4_data[v3v4_data["type"].str.startswith("(full)")]  ### 22897
    partial_data = v3v4_data[~v3v4_data["type"].str.startswith("(full)")] ### 18868
info_df = pd.read_csv(os.path.join(my_data_path, '16S_rRNA_annotation_table.tsv'),sep="\t",header=None)
info_df.rename(columns={0:"id", 4:"label_name", 5:"GTDB_lineage", 7:"pos", 10:"partial_percent"},inplace=True)
info_full_df = info_df[["id", "label_name", "GTDB_lineage", "pos", "partial_percent"]].merge(full_data[["id", "sequence"]],how="right",on="id")
info_partial_df = info_df[["id", "label_name", "GTDB_lineage", "pos", "partial_percent"]].merge(partial_data[["id", "sequence"]],how="right",on="id")

if label_type=="genus":
    info_full_df["label_name"] = info_full_df["GTDB_lineage"].apply(lambda x:";".join(x.split(";")[:-1]))
    info_partial_df["label_name"] = info_partial_df["GTDB_lineage"].apply(lambda x:";".join(x.split(";")[:-1]))
    
    print("---",info_full_df.__len__(), info_full_df["label_name"].nunique())
    print("---",info_partial_df.__len__(), info_partial_df["label_name"].nunique())

    info_all_df = pd.concat([info_full_df, info_partial_df])
    print("---",info_all_df.__len__(), info_all_df["label_name"].nunique())
    info_full_df = info_full_df[~info_full_df["label_name"].str.endswith("g__")]
    info_partial_df = info_partial_df[~info_partial_df["label_name"].str.endswith("g__")]

if data_type=="full":
    item_df = info_full_df
elif data_type=="partial":
    item_df = info_partial_df
elif data_type=="all":
    item_df = pd.concat([info_full_df, info_partial_df])

# no dup
import statistics
t = item_df['sequence'].str.len().to_numpy()
print(np.min(t), np.max(t), statistics.median(t))
print(np.mean(t), np.std(t, ddof=1))
print(info_full_df['label_name'].nunique())
# pdb.set_trace()

# ==================== dedup
print(item_df.__len__(), item_df["label_name"].nunique())
item_dedup_df = item_df.drop_duplicates(subset=["label_name", "sequence"])
print(item_dedup_df.__len__(), item_dedup_df["label_name"].nunique())
error = 0
seq_list = []
for gp in item_dedup_df.groupby("sequence"):
    if gp[1]["label_name"].nunique()>1:
        error+=1
        seq_list.append(gp[0])

drop_dup_match_error = item_dedup_df[item_dedup_df["sequence"].apply(lambda x:x in seq_list)]
seq_no_dict = {}
n=0
for i in drop_dup_match_error["sequence"].value_counts().index:
    seq_no_dict[i] = n
    n+=1
drop_dup_match_error["seq_number"] = drop_dup_match_error["sequence"].map(seq_no_dict)
item_df = item_dedup_df[item_dedup_df["sequence"].apply(lambda x:x not in seq_list)]
print(item_df.__len__(), item_df["label_name"].nunique())
use_label_lst = item_df["label_name"].value_counts()[item_df["label_name"].value_counts()>=cutoff].index
item_df = item_df[item_df["label_name"].apply(lambda x:x in use_label_lst)]
print(item_df.__len__(), item_df["label_name"].nunique())  # genus 9629, 156; species 7816 193

# dup
print("-----final-----")
import statistics
t = item_df['sequence'].str.len().to_numpy()
print(np.min(t), np.max(t), statistics.median(t))
print(np.mean(t), np.std(t, ddof=1))

# item_df.to_csv('%s_%s.csv'%(label_type, data_type), index=None)

# ====================

obj = kmer_featurization(k) # initialize a kmer_featurization object
item_df["sequence_kmer"] = item_df["sequence"].str.replace("N", "")
item_df.reset_index(drop=True,inplace=True)

# all_seq_lst = item_df['sequence'].to_list()
# for i in range(len(all_seq_lst)):
#     if 'N' in all_seq_lst[i]:
#         print(all_seq_lst[i])

max_seq_len = max([len(x) for x in item_df["sequence"]])
bases = "NTCGA" # NTCGA
tcga_index = {base:i for i, base in enumerate(bases)}
tcga_dict = {}
for tcga in tcga_index.keys():
    onehot_list = np.zeros(len(tcga_index.keys()))
    onehot_list[tcga_index[tcga]] = 1
    tcga_dict[tcga] = onehot_list # {'T': array([0., 1., 0., 0., 0.]),...}

def seed_everything(seed=688):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True    
seed_everything(seed=global_seed)

class RNA16SDataset:
    def __init__(self, features, targets, seqs_list, max_seq_len):
        self.features = features
        self.targets = targets
        self.generate_onehot_mat(seqs_list, max_seq_len)
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float),
            'onehot_mat':torch.tensor(self.onehot_mat[idx,:],dtype=torch.float)
        }
        return dct

    def generate_onehot_mat(self,seqs_list,max_seq_len):
        onehot_mat = []
        for seqs in seqs_list:
            seq_mat = []
            for seq in seqs:
                seq_mat.append(tcga_dict[seq])
            if len(seqs)<max_seq_len:
                for i in range(len(seqs),max_seq_len):
                    seq_mat.append(np.array([0.2]*len(bases)))
            seq_mat = np.array(seq_mat)
            onehot_mat.append(seq_mat)
        self.onehot_mat = np.array(onehot_mat)

def iteration(model, optimizer, loss_fn, dataloader, device, is_training = False):
    if is_training:
        model.train()
    else:
        model.eval()
    final_loss = 0
    all_pred = []
    all_label = []
    for batch_data in dataloader:
        seq_mat, seq_mer, targets = batch_data["onehot_mat"].to(device), batch_data['x'].to(device), batch_data['y'].to(device)
        outputs = model(seq_mat, seq_mer)
        if is_training:
            loss = loss_fn(outputs, targets.argmax(dim=1))
            reg_loss = 0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    reg_loss += torch.norm(param, p=2)
            loss += LAMBDA_L2 * reg_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss += loss.item()
        # print(outputs.shape, targets.shape)
        # pdb.set_trace()
        outputs = outputs.sigmoid().detach().to('cpu')
        outputs = torch.max(outputs, 1)[1].numpy()
        targets = targets.detach().to('cpu')
        targets = torch.max(targets, 1)[1].numpy() 
        all_pred.append(outputs)
        all_label.append(targets)

    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
 
    accuracy = accuracy_score(all_label, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(all_label, all_pred, average='macro', zero_division = 0)
    res = [accuracy, precision, recall, f1]
    if is_training:
        final_loss /= len(dataloader)
        res += [final_loss]
    # else:
        # np.save('model_var/%s_pred'%logname, all_pred)
        # np.save('model_var/%s_label'%logname, all_label)
    return res

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    for data in dataloader:
        onehot_mat, inputs = data["onehot_mat"].to(device), data['x'].to(device)
        with torch.no_grad():
            outputs = model(onehot_mat,inputs)  
        preds.append(outputs.sigmoid().detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds

def start(train_index, valid_index, fold, seed, dropout):
    train_X = kmer_X[train_index]
    train_Y = true_Y[train_index]
    valid_X = kmer_X[valid_index]
    valid_Y = true_Y[valid_index]
    if len(train_X)%BATCH_SIZE==1: # drop last
        train_X = train_X[:-1]
        train_Y = train_Y[:-1]

    train_dataset = RNA16SDataset(train_X, train_Y, item_df.loc[train_index,"sequence"],max_seq_len)
    valid_dataset = RNA16SDataset(valid_X, valid_Y, item_df.loc[valid_index,"sequence"],max_seq_len)
    # for batch_data in train_dataset:
    #     seq_mat, seq_mer, targets =  batch_data["onehot_mat"], batch_data['x'], batch_data['y']
    #     break
    # import pdb
    # pdb.set_trace()
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    seq_mat_dim = train_dataset.onehot_mat.shape[-1]
    seq_mer_dim = train_X.shape[1]
    n_targets = train_Y.shape[1]

    model = RNA_Model(
        use_model = use_model,
        seq_mat_dim = seq_mat_dim,
        seq_mer_dim = seq_mer_dim,
        n_targets = n_targets,
        hidden_size = hidden_size,
        dropout = dropout,
    )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model.to(DEVICE)
    else:
        model.to(DEVICE)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #  weight_decay=WEIGHT_DECAY
    loss_fn = nn.CrossEntropyLoss()
    
    early_step = 0
    best_epoch, best_acc =0, 0
    print(f'Fold:{fold}')
    save_model_file = osp.join(save_model_path, logname+"_seed="+str(seed)+"_f"+str(fold)+".pth")
    for epoch in range(1, EPOCHS+1):
        res_lst = iteration(model, optimizer, loss_fn, trainloader, DEVICE, is_training = True)
        train_acc, train_p, train_r, train_f1, train_loss = res_lst

        res_lst = iteration(model, optimizer, loss_fn, validloader, DEVICE)
        valid_acc, valid_p, valid_r, valid_f1 = res_lst
    
        print('Epoch:{:03d}; Loss:{:.4f}; ACC:{:.4f},{:.4f};F1:{:.4f},{:.4f}; P:{:.4f},{:.4f}; R:{:.4f},{:.4f}'.format(
            epoch, train_loss,
            train_acc, valid_acc, train_f1, valid_f1,
            train_p, valid_p, train_r, valid_r))
   
        if epoch == EPOCHS:
            print('Best Epoch:{:03d}; ACC:{:.4f}'.format(best_epoch, best_acc))
        if valid_acc >= best_acc:
            best_epoch = epoch
            best_acc = valid_acc
            if EPOCHS > 20:  
                # save_model_file = osp.join(save_model_path, logname+"_seed="+str(seed)+"_f"+str(fold)+".pth")
                torch.save(model.state_dict(), save_model_file)
        elif(EARLY_STOP == True):  
            early_step += 1
            if (early_step >= EARLY_STOPPING_STEPS):
                break
    # ========== 
    testloader = validloader
    test_model = RNA_Model(
        use_model = use_model,
        seq_mat_dim = seq_mat_dim,
        seq_mer_dim = seq_mer_dim,
        n_targets = n_targets,
        hidden_size = hidden_size,
        dropout = dropout,
    )
    test_model.load_state_dict(torch.load(save_model_file))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        test_model = nn.DataParallel(test_model)
        test_model.to(DEVICE)
    else:
        test_model.to(DEVICE)
    
    pred_Y = np.zeros((len(valid_X), n_targets))
    pred_Y = inference_fn(test_model, testloader, DEVICE)
    pred_y = np.argmax(pred_Y,axis=1)
    test_y = np.argmax(valid_Y,axis=1)

    test_acc = accuracy_score(test_y, pred_y)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(test_y, pred_y, average='macro', zero_division = 0)
    print('ACC:{:.4f};F1:{:.4f}; P:{:.4f}; R:{:.4f}'.format(
        test_acc, test_f1, test_p, test_r)
    )
    return pred_Y # for seeds mean

if __name__ == '__main__':
    kmer_save_path = osp.join(my_data_path, f"kmer_X_{data_type}_{seq_type}_cut{cutoff}_{k}mer_{label_type}")
    if os.path.exists(kmer_save_path+'.npy'):
        print("load kmer", kmer_save_path)
        kmer_X = np.load(kmer_save_path+'.npy',allow_pickle=True)
    else:
        print("save kmer",kmer_save_path)
        kmer_X = obj.obtain_kmer_feature_for_a_list_of_sequences(item_df["sequence_kmer"], write_number_of_occurrences=False)
        np.save(kmer_save_path, kmer_X)
    print("check kmer shape:",kmer_X.shape)
    
    # OneHotEncoder 如果数据集不包含全部会出问题
    label_encoder = OneHotEncoder(sparse=False)
    clf = label_encoder.fit(item_df["label_name"].values.reshape(-1,1))
    true_Y = clf.transform(item_df["label_name"].values.reshape(-1,1))

    # df = item_df["label_name"].to_frame()
    # df.columns = ['Label_Name']
    # df['Label'] = true_Y.argmax(axis=1)
    # df = df.drop_duplicates()
    # print(len(df))
    # df.to_csv("%s_%s.csv" % (label_type, data_type),index=None)


    pred_seeds = np.zeros((item_df.shape[0], 1))
    for seed in kfold_seeds:
        if kfold_type == 'KFold':
            kf = KFold(n_splits=n_splits, shuffle = True, random_state=seed)
        if kfold_type == 'StratifiedKFold':
            kf = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=seed)
        preds = pd.DataFrame()
        fold = 1
        for train_index,test_index in kf.split(item_df.index, item_df["label_name"]):
            # slurm_split_path = 'slurm_split_data'
            # if label_type=='species':
            #     save_front_name = 'rs_mer=7_CNN+MLP_fold=%s_cut=%s_%s_t_' % (n_splits, cutoff, data_type)
            # else:
            #     save_front_name = 'rg_mer=7_CNN+MLP_fold=%s_cut=%s_%s_t_' % (n_splits, cutoff, data_type)
            pred_tmp = start(train_index, test_index, fold, seed, dropout)
            pred_tmp = pd.DataFrame(pred_tmp)
            fold+=1
            pred_tmp.index = test_index
            preds = pd.concat([preds, pred_tmp])

        a_pred_y = preds.loc[item_df.index].values.argmax(axis=1)
        a_label_y = true_Y[item_df.index].argmax(axis=1)
        test_acc = accuracy_score(a_label_y, a_pred_y)
        test_p, test_r, test_f1, _ = precision_recall_fscore_support(a_label_y, a_pred_y, average='macro', zero_division = 0)
        print('Seed:{}; ACC:{:.4f}; F1:{:.4f}; P:{:.4f}; R:{:.4f}'.format(
            seed, test_acc, test_f1, test_p, test_r)
        )
 
        pred_seeds += preds
        pred_seeds /= len(kfold_seeds)
    pred_y = pred_seeds.loc[item_df.index].values.argmax(axis=1)
    label_y = true_Y[item_df.index].argmax(axis=1)
    test_acc = accuracy_score(label_y, pred_y)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(label_y, pred_y, average='macro', zero_division = 0)
    print('Mean of seeds:{}; ACC:{:.4f}; F1:{:.4f}; P:{:.4f}; R:{:.4f}'.format(
        kfold_seeds, test_acc, test_f1, test_p, test_r)
    )
    
    
