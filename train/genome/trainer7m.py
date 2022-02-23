import os
import sys
import pdb
import copy
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasetG7m import TorchDataset
from model import MLPModel, ResidualModel
# ge5_le50_f5
parser = argparse.ArgumentParser(description='')
parser.add_argument('--log', type=str, help='must have logname')
parser.add_argument('--model', type=str, default='ResNN', help='MLP, ResNN')
current_path = os.path.dirname(os.path.realpath(__file__))
log_dir = 'log'
if not osp.exists(log_dir):
    os.mkdir(log_dir)
model_save_dir = 'model_params'
if not osp.exists(model_save_dir):
    os.mkdir(model_save_dir)
    
param_dict = {
    'has_le':True,
    'min_sample_num':5,
    'max_sample_num':50,
    # train
    'seed':0,
    'epochs':5, # 2000
    'batch_size':1024, # 256 
    'lr':1e-3,
    'dropout':0.3,
    'h_dim':256,
    'lambda_l1':0,
    'lambda_l2':0,
    # 'patience':30,   
    'n_fold':5,
    'save_model':True, # True
    # data
    'label_type':'species', # species genus
    'train_size':0.8,
    'use_test':False,
    'shuffle':True,
    'stratify':True, # False
    'reprocess':False,
}  

# =======================================================
# from tensorboardX import SummaryWriter
# logger = SummaryWriter(log_dir="log")
class Model_Trainer(object):
    def __init__(self, **param_dict):
        for (key, value) in param_dict.items():
            setattr(self, key, value)

        self.setup_seed(self.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = TorchDataset(**param_dict)
        param_dict['n_label'] = self.dataset.n_label
        param_dict['ft_dim'] = self.dataset.kmer_mat.shape[1]
        # print("n_label:",param_dict['n_label'], 'n_feat:',param_dict['ft_dim'])
        self.param_dict = param_dict
        self.build_model()
        self.all_fold_res_list = []

    def build_model(self):
        if self.model == 'MLP':
            self.model = MLPModel(**self.param_dict).to(self.device)
        elif self.model == 'ResNN':
            self.model = ResidualModel(**self.param_dict).to(self.device)
        # self.model = CNNModel(**self.param_dict).to(self.device)
        # self.model = TabNetModel(**self.param_dict).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.best_res = None
        self.min_dif = -float('inf')

    def setup_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
    @staticmethod
    def print_res(res_list, best_res = False):
        epoch, train_acc, train_p, train_r, train_f1, \
            valid_acc, valid_p, valid_r, valid_f1, \
            test_acc, test_p, test_r, test_f1 = res_list

        msg_log = 'Epoch:{:03d}; ACC:{:.4f},{:.4f},{:.4f}; P:{:.4f},{:.4f},{:.4f}; '\
            'R:{:.4f},{:.4f},{:.4f}; F1:{:.4f},{:.4f},{:.4f}'.format(epoch,
                train_acc, valid_acc, test_acc,
                train_p, valid_p, test_p,
                train_r, valid_r, test_r,
                train_f1, valid_f1, test_f1,
            )
        if best_res:
            msg_log = 'Best '+ msg_log
        print(msg_log)
    
    def start_iteration(self, dataloader, epoch, is_training=False):
        if is_training:
            self.model.train()
            return self.iteration(dataloader, epoch, is_training)
        else:
            self.model.eval()
            with torch.no_grad():
                return self.iteration(dataloader, epoch, is_training)

    def iteration(self, dataloader, epoch, is_training=False):
        all_loss = []
        all_pred = []
        all_label = []
        train_idx1 = []
        test_idx1 = []
        for item_idx, label, data in dataloader:
            label = label.to(self.device)
            data = data.to(self.device)
            pred = self.model(data)
            # print(data.shape)
            if is_training:
                train_idx1.append(data.to('cpu').numpy())
                loss = self.loss(pred, label)
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)
                loss += self.lambda_l1 * param_l1_loss
                loss += self.lambda_l2 * param_l2_loss 
                all_loss.append(loss.detach().to('cpu'))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                test_idx1.append(data.to('cpu').numpy())
            pred = pred.detach().to('cpu')
            pred = torch.max(pred, 1)[1].numpy()
            label = label.detach().to('cpu').numpy()
            all_pred.append(pred)
            all_label.append(label)
        
        
        if is_training:
            # train_idx1 = np.concatenate(train_idx1, axis=0)
            # np.save('train_idx1',train_idx1.reshape(-1,4**7))
            print("loss: {:.4f};".format(np.mean(np.array(all_loss))), end = ' ')
            # logger.add_scalar("train loss", np.mean(np.array(all_loss)), global_step=epoch)
        else:
            # test_idx1 = np.concatenate(test_idx1, axis=0)
            # np.save('test_idx1',test_idx1.reshape(-1,4**7))
            # exit()
            pass

        all_pred = np.concatenate(all_pred, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        # if not is_training:
        #     np.save('temp/%s_pred'%self.logname, all_pred)
        #     np.save('temp/%s_label'%self.logname, all_label)
        accuracy = accuracy_score(all_label, all_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(all_label, all_pred, average='macro', zero_division = 0)
        res = [accuracy, precision, recall, f1]
        return list(res)

    def start(self, fold_num=0, display=True):
        self.dataset.train_sampler = SubsetRandomSampler(self.dataset.train_index)
        self.dataset.valid_sampler = SubsetRandomSampler(self.dataset.valid_index)
        # self.dataset.test_sampler = SubsetRandomSampler(self.dataset.test_index)
        train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=self.dataset.train_sampler)
        valid_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=self.dataset.valid_sampler)
        # test_loader  = DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=self.dataset.test_sampler )
        for epoch in range(1, self.epochs+1):
            # st = time.time()
            res_list =  self.start_iteration(train_loader, epoch, is_training=True )
            res_list += self.start_iteration(valid_loader, epoch, is_training=False)
            # res_list += self.iteration( test_loader, epoch, is_training=False)
            res_list += res_list[-4:] # for print format
            # logger.add_scalar("train ACC", res_list[0], global_step=epoch)
            # logger.add_scalar("train F1", res_list[3], global_step=epoch)
            # for name, param in self.model.named_parameters():
            #         logger.add_histogram(name, param.data.to('cpu').numpy(), global_step=epoch)
            valid_f1 = res_list[2*4-1]

            # early_stopping = EarlyStopping(patience=self.patience, verbose=False)
            # early_stopping(valid_r2, self.model)
            # if self.early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            res_list = [epoch]+res_list
            if valid_f1 > self.min_dif:
                self.min_dif = valid_f1
                self.best_res = copy.copy(res_list)
                if self.save_model:
                    torch.save(self.model.state_dict(), self.save_model_path+'_besk.pkl')
                
            if display:
                self.print_res(res_list, best_res = False)

            if epoch % 20 == 0 or epoch == self.epochs:
                self.print_res(self.best_res, best_res = True)
            # print(time.time()-st)
            if epoch == self.epochs and self.save_model:
                torch.save(self.model.state_dict(), self.save_model_path+'_last.pkl')
        return self.best_res

    def k_fold(self):
        if self.stratify:
            kf = StratifiedKFold(n_splits = self.n_fold, shuffle = True, random_state = self.seed)
        else:
            kf = KFold(n_splits = self.n_fold, shuffle = True, random_state = self.seed)
        fold_num = 1
        index_vec = np.arange(0, self.dataset.n_item, 1)
        for train_index, valid_index in kf.split(index_vec, self.dataset.label_vec):
            self.save_model_name = '{}_seed{}_f{}'.format(self.logname, self.seed, fold_num)
            self.save_model_path = '{}/{}'.format(model_save_dir, self.save_model_name)
            train_df = pd.DataFrame()
            test_df = pd.DataFrame()
            train_df[str(self.seed)+'_'+str(fold_num)] = train_index
            test_df[str(self.seed)+'_'+str(fold_num)] = valid_index
            train_df.to_csv(osp.join(model_save_dir, self.logname+"_train"+str(self.seed)+'_'+str(fold_num)+".csv"))
            test_df.to_csv(osp.join(model_save_dir, self.logname+"_test"+str(self.seed)+'_'+str(fold_num)+".csv"))

            self.build_model()
            self.dataset.train_index = train_index
            self.dataset.valid_index = valid_index
            self.dataset.test_index = valid_index
            fold_best_res = self.start(fold_num)
            self.all_fold_res_list.append(fold_best_res)
            fold_num += 1

        # save res
        res_df = pd.DataFrame(self.all_fold_res_list)
        res_df.columns = ['epoch','train_acc','train_p','train_r','train_f1',\
            'valid_acc','valid_p','valid_r','valid_f1',\
            'test_acc','test_p','test_r','test_f1']
        res_df_path = os.path.join(current_path, log_dir,
                               self.save_model_name[:-4] + '.csv')
        res_df.to_csv(res_df_path)
# python trainer.py --log ge10_le50_f5_r_test
if __name__ == '__main__':
    args = parser.parse_args() 
    logname = args.log
    param_dict.update({'logname':logname})
    param_dict.update({'model':args.model})

    if ('le' in logname):
        param_dict.update({'has_le': True})
    else:
        param_dict.update({'has_le': False})

    trainer = Model_Trainer(**param_dict)

    if param_dict['n_fold']:
        assert ('f5' in logname) == (param_dict['n_fold']==5)
        trainer.k_fold()
    else:
        trainer.start()