import os
import pdb
import time
import numpy as np
import pandas as pd
import os.path as osp
from sklearn.model_selection import train_test_split
# ===== base path
current_path = os.path.dirname(os.path.realpath(__file__))
temp_data_path = osp.join(current_path, 'data')
data_base_path = '/data/xbiome/origin'
kmer_path = 'genome_kmer'
# ===== file name
new_info_file = osp.join(temp_data_path, 'info_genome_ge5_le50_dedup.csv') # ============================ !!!
new_info_df = pd.read_csv(new_info_file)

def select_datapath(min_sample_num, max_sample_num, has_le=False):
    use_species_path = osp.join(temp_data_path, 'species_ge%d_dedup.csv'% min_sample_num)
    if not has_le:
        use_genome_path = osp.join(temp_data_path, 'genome_ge%d_dedup.csv'% min_sample_num)
        new_kmer_path = osp.join(temp_data_path, 'kmer7_mat_ge%d_dedup.npy'% min_sample_num)
    else:
        use_genome_path = osp.join(temp_data_path, 'genome_ge%d_le%d_dedup.csv'% (min_sample_num, max_sample_num))
        new_kmer_path = osp.join(temp_data_path, 'kmer7_mat_ge%d_le%d_dedup.npy'% (min_sample_num, max_sample_num))
    return use_species_path, use_genome_path, new_kmer_path

# ===== read file
'''
ge5_le50  50584 1876
ge10_le50 47284 1375
ge30_le50 37296 786
'''

def get_species_df(new_info_df, use_species_path, 
                   min_sample_num, reprocess=False):
    if os.path.exists(use_species_path) and reprocess == False:
        return pd.read_csv(use_species_path)
    count_df = new_info_df.groupby('MGnify_accession').count()
    count_df = count_df['Genome'].reset_index()
    count_df = count_df.sort_values('Genome',ascending=False)
    use_species_df = count_df[count_df['Genome']>=min_sample_num]
    use_species_df = use_species_df.rename(columns={'Genome':'Count'})

    use_species_df.to_csv(use_species_path, index=None)
    return use_species_df # 输出为按照Count排序的，决定了label编码

def get_genera_df(use_species_df, new_info_df):
    genera_df = new_info_df[['MGnify_accession','Genus']].copy()
    use_genera_df = pd.merge(use_species_df, genera_df, on='MGnify_accession', how='left')
    return use_genera_df

def get_genome_df(new_info_df, use_species_df, use_genome_path, 
                  min_sample_num, has_le, 
                  reprocess=False):
    if os.path.exists(use_genome_path) and reprocess == False:
        return pd.read_csv(use_genome_path)

    info_df = new_info_df[new_info_df['MGnify_accession'].isin(use_species_df['MGnify_accession'])]
    # ========== ge
    if not has_le:
        use_genome_df = info_df.groupby('MGnify_accession',group_keys=False).apply(lambda x: x.sample(n=min_sample_num, random_state=42)) # 一定注意随机种子！
        use_genome_df = use_genome_df.sort_index()

    # ========== ge + le
    else:
        info_df = info_df.rename(columns={'Count':'Count_ori'}) # 避免重名
        use_genome_df = pd.merge(info_df, use_species_df, on='MGnify_accession').sort_index()

    use_genome_df.to_csv(use_genome_path, index_label='Index')
    return use_genome_df 

class GenomeDataset(object):
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        use_species_path, use_genome_path, self.new_kmer_path = select_datapath(self.min_sample_num,  self.max_sample_num, has_le=self.has_le)
        self.use_species_df = get_species_df(new_info_df, use_species_path, self.min_sample_num, self.reprocess)
        self.use_genome_df = get_genome_df(new_info_df, self.use_species_df, use_genome_path, self.min_sample_num, self.has_le, self.reprocess)
        self.n_item = self.use_genome_df.__len__()
        # self.n_label = self.use_species_df.__len__()
        # print(self.n_item, self.n_label)
        if self.label_type == 'species':
            self.label_colname = 'MGnify_accession'
            self.n_label = self.use_species_df.__len__()
            self.label_dict = self.use_species_df[self.label_colname].unique().tolist()
        elif self.label_type == 'genus':
            self.label_colname =  'Genus'
            self.use_genera_df = get_genera_df(self.use_species_df, new_info_df)
            self.label_dict = self.use_genera_df[self.label_colname].unique().tolist()
            self.n_label = self.label_dict.__len__()
        print(self.n_item, self.n_label)
        self.label_vec = None
        self.kmer_mat = np.zeros((self.n_item, 4**7), dtype=np.float32)
        self.error_genome_lst = []

        self.generate_data()
        
    def generate_data(self,):
        # ========== label
        label_dict = self.label_dict
        # label_dict = self.use_species_df[self.label_colname].unique().tolist()
        self.use_genome_df['label'] = self.use_genome_df[self.label_colname].apply(lambda x : label_dict.index(x))
        self.label_vec = self.use_genome_df['label']


        # pdb.set_trace()
        # ========== data
        time1 = time.time()
        if os.path.exists(self.new_kmer_path) and self.reprocess==False:
            self.kmer_mat = np.load(self.new_kmer_path, allow_pickle = True)
        else: 
            for i in range(len(self.use_genome_df)):
                row = self.use_genome_df.iloc[i]
                genome_name = row['Genome']
                species = row['MGnify_accession']
                try:
                    self.kmer_mat[i] = np.load(osp.join(data_base_path, kmer_path, genome_name+'.npy'), allow_pickle = True) 
                except:
                    self.error_genome_lst.append(genome_name)
                    continue
                if i%1000==0:
                    print(i)

            self.kmer_mat = self.kmer_mat.astype('float32')
            np.save(self.new_kmer_path, self.kmer_mat)
            time2 = time.time()
            print(time2-time1)

        # assert self.kmer_mat.shape[0] == self.use_genome_df.__len__()
        # for check_idx in range(self.kmer_mat.shape[0]):
        #     # check_idx = 11111
        #     genome_info = self.use_genome_df.iloc[check_idx]
        #     genome_name = genome_info['Genome']
    
        #     out_kmer = self.kmer_mat[check_idx]
        #     species = genome_info['MGnify_accession']
        #     ori_kmer = np.load(osp.join(data_base_path, kmer_path, genome_name+'.npy'))
        #     assert (ori_kmer == out_kmer).all()
        #     if check_idx % 1000 ==0:
        #         print(check_idx)
        # print('exit')
        # exit()

        # ========== split
        if self.stratify:
            self.split_index(self.label_vec)
        else:
            self.split_index(None)
    
        
    def split_index(self, stratify):
        self.train_index, self.valid_index = train_test_split(
            np.arange(0,self.n_item,1), train_size=self.train_size, random_state=self.seed, shuffle=self.shuffle, stratify=stratify)
        self.test_index = self.valid_index

        if self.use_test:
            if stratify is not None:
                valid_stratify = stratify[self.valid_index] # 顺序同self.valid_index
            else: 
                valid_stratify = stratify
            self.valid_index, self.test_index = train_test_split(
                self.valid_index, train_size=0.5, random_state=self.seed, shuffle=self.shuffle, stratify=valid_stratify)

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
class TorchDataset(Dataset, GenomeDataset):
    def __init__(self, **kwargs):
        GenomeDataset.__init__(self,**kwargs)
        Dataset.__init__(self)

    def __getitem__(self, item_index):
        label = self.label_vec[item_index]
        data = self.kmer_mat[item_index] 
        return item_index, label, data
    def __len__(self):
        return self.n_item

if __name__ == '__main__':
    param_dict = {
        'has_le':True,
        'min_sample_num':5,
        'max_sample_num':50,
        'seed':42,
        'label_type':'species',
        'train_size':0.8,
        'use_test':False,
        'shuffle':True,
        'stratify':True,
        'reprocess':False,
    }
    dataset = GenomeDataset(**param_dict)
    print(dataset.kmer_mat.dtype, dataset.kmer_mat.shape)
    print(dataset.label_vec.dtype,dataset.label_vec.shape)
    print(dataset.train_index.__len__(),dataset.valid_index.__len__(),dataset.test_index.__len__())

    batch_size = 4
    dataset = TorchDataset(**param_dict)
    dataset.train_sampler = SubsetRandomSampler(dataset.train_index)
    dataset.valid_sampler = SubsetRandomSampler(dataset.valid_index)
    dataset.test_sampler  = SubsetRandomSampler(dataset.test_index )
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=dataset.train_sampler)
    valid_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=dataset.valid_sampler)
    test_loader  = DataLoader(dataset=dataset, batch_size=batch_size, sampler=dataset.test_sampler )
    for i, label, data in train_loader:
        print(label.dtype, label.shape)
        print(data.dtype, data.shape)
        break
    pdb.set_trace()

    # check = False # 检查对齐
    # if check:
    #     i0 = [0,1000,3000,10000]
    #     outx = dataset.kmer_mat[i0]
    #     outy = dataset.label_vec[i0]
    #     use_df = dataset.use_genome_df.iloc[i0]
    #     print(use_df[['MGnify_accession','label']]) 
    #     print((np.array(use_df['label'].to_list())==outy).all())

    #     data_base_path = '/data/未知君/origin'
    #     kmer_path = 'genome_kmer'
    #     for i in range(len(use_df)):
    #         g = use_df['Genome'].iloc[i]
    #         inx = np.load(osp.join(data_base_path, kmer_path, g+'.npy'),allow_pickle=True)
    #         iny = use_df[['MGnify_accession','label']].iloc[i]
    #         print(g, (np.array(inx)==outx[i]).all()) 

    #     train_idx = set(datasetG.train_index)
    #     valid_idx = set(datasetG.valid_index)
    #     # test_idx = set(datasetG.test_index)
    #     len(train_idx|valid_idx)==len(train_idx)+len(valid_idx)