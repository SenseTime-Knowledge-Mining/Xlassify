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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--log', type=str, help='must have logname')
parser.add_argument('--use_model', type=str, default='MLP', help='MLP,CNN, CNN+MLP')
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
    info_full_df = info_full_df[~info_full_df["label_name"].str.endswith("g__")]
    info_partial_df = info_partial_df[~info_partial_df["label_name"].str.endswith("g__")]

if data_type=="full":
    item_df = info_full_df
elif data_type=="partial":
    item_df = info_partial_df
elif data_type=="all":
    item_df = pd.concat([info_full_df, info_partial_df])
import statistics
t = item_df['sequence'].str.len().to_numpy()
print(np.min(t), np.max(t), statistics.median(t))
print(np.mean(t), np.std(t, ddof=1))
# pdb.set_trace()

# ==================== dedup
item_dedup_df = item_df.drop_duplicates(subset=["label_name", "sequence"])

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

use_label_lst = item_df["label_name"].value_counts()[item_df["label_name"].value_counts()>=cutoff].index
item_df = item_df[item_df["label_name"].apply(lambda x:x in use_label_lst)]
print(item_df.__len__(), item_df["label_name"].nunique())  # genus 9629, 156; species 7816 193

import statistics
t = item_df['sequence'].str.len().to_numpy()
print(np.min(t), np.max(t), statistics.median(t))
print(np.mean(t), np.std(t, ddof=1))
# pdb.set_trace()
# ====================

obj = kmer_featurization(k) # initialize a kmer_featurization object
item_df["sequence_kmer"] = item_df["sequence"].str.replace("N", "")
item_df.reset_index(drop=True,inplace=True)

# all_seq_lst = item_df['sequence'].to_list()
# for i in range(len(all_seq_lst)):
#     if 'N' in all_seq_lst[i]:
#         print(all_seq_lst[i])
# pdb.set_trace()
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

import joblib
from sklearn.naive_bayes import MultinomialNB
def start_mnb(train_index, valid_index, fold, seed):
    print(f'Fold:{fold}')
    train_X = kmer_X[train_index]
    train_Y = true_Y[train_index]
    valid_X = kmer_X[valid_index]
    valid_Y = true_Y[valid_index]
    # to 0-1
    train_X = np.where(train_X>0, 1, 0)
    valid_X = np.where(valid_X>0, 1, 0)
    
    train_y = np.argmax(train_Y, axis=1)
    test_y = np.argmax(valid_Y, axis=1)

    mnb = MultinomialNB()

    clf  = mnb.fit(train_X, train_y)

    pred_y = clf.predict(valid_X)
    
    test_acc = accuracy_score(test_y, pred_y)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(test_y, pred_y, average='macro', zero_division = 0)
    print('ACC:{:.4f};F1:{:.4f}; P:{:.4f}; R:{:.4f}'.format(
        test_acc, test_f1, test_p, test_r)
    )
    # save_model_file = osp.join(save_model_path, logname+"_seed="+str(seed)+"_f"+str(fold)+".model")
    # joblib.dump(clf, save_model_file)
    return pred_y

if __name__ == '__main__':
    kmer_save_path = osp.join(my_data_path, f"kmer_X_{data_type}_{seq_type}_cut{cutoff}_{k}mer_{label_type}")
    if os.path.exists(kmer_save_path+'.npy'):
        print("load kmer", kmer_save_path)
        kmer_X = np.load(kmer_save_path+'.npy',allow_pickle=True)
    else:
        print("save kmer",kmer_save_path)
        kmer_X = obj.obtain_kmer_feature_for_a_list_of_sequences(item_df["sequence_kmer"], write_number_of_occurrences=False)
        np.save(kmer_save_path, kmer_X)
 
    # OneHotEncoder 如果数据集不包含全部会出问题
    label_encoder = OneHotEncoder(sparse=False)
    clf = label_encoder.fit(item_df["label_name"].values.reshape(-1,1))
    true_Y = clf.transform(item_df["label_name"].values.reshape(-1,1))
    test_accs, test_ps, test_rs, test_f1s = [], [], [], []
    pred_seeds = np.zeros((item_df.shape[0], 1))
    for seed in kfold_seeds:
        if kfold_type == 'KFold':
            kf = KFold(n_splits=n_splits, shuffle = True, random_state=seed)
        if kfold_type == 'StratifiedKFold':
            kf = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=seed)
        preds = pd.DataFrame()
        fold = 1
        for train_index,test_index in kf.split(item_df.index, item_df["label_name"]):
            slurm_split_path = 'slurm_split_data'
            if label_type=='species':
                save_front_name = 'rs_mer=7_CNN+MLP_fold=%s_cut=%s_%s_t_' % (n_splits, cutoff, data_type)
            else:
                save_front_name = 'rg_mer=7_CNN+MLP_fold=%s_cut=%s_%s_t_' % (n_splits, cutoff, data_type)
            train_index = pd.read_csv(osp.join(slurm_split_path, save_front_name+"train"+str(seed)+'_'+str(fold)+".csv")).iloc[:,1].to_list()
            test_index = pd.read_csv(osp.join(slurm_split_path, save_front_name+"test"+str(seed)+'_'+str(fold)+".csv") ).iloc[:,1].to_list()
            pred_tmp = start_mnb(train_index, test_index, fold, seed)
            pred_tmp = pd.DataFrame(pred_tmp)
            fold+=1
            pred_tmp.index = test_index
            preds = pd.concat([preds, pred_tmp])
            
        a_pred_y = preds.loc[item_df.index].values#.argmax(axis=1)
        a_label_y = true_Y[item_df.index].argmax(axis=1)
        test_acc = accuracy_score(a_label_y, a_pred_y)
        test_p, test_r, test_f1, _ = precision_recall_fscore_support(a_label_y, a_pred_y, average='macro', zero_division = 0)
        print('Seed:{}; ACC:{:.4f}; F1:{:.4f}; P:{:.4f}; R:{:.4f}'.format(
            seed, test_acc, test_f1, test_p, test_r)
        )
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_ps.append(test_p)
        test_rs.append(test_r)
        

    # pred_y = pred_seeds.loc[item_df.index].values.reshape(-1)#.argmax(axis=1)
    # label_y = true_Y[item_df.index].argmax(axis=1)
    # test_acc = accuracy_score(label_y, pred_y)
    # test_p, test_r, test_f1, _ = precision_recall_fscore_support(label_y, pred_y, average='macro', zero_division = 0)
    print('Mean of seeds:{}; ACC:{:.4f}; F1:{:.4f}; P:{:.4f}; R:{:.4f}'.format(
        kfold_seeds, np.mean(test_accs), np.mean(test_f1s), np.mean(test_ps), np.mean(test_rs))
    )
 
