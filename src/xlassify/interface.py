# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np
import pandas as pd
import os.path as osp
import logging.handlers
from .tools import kmer_utils
from .network.model import ResidualModel, RNA_Model

import requests
from tqdm import tqdm

__all__ = [
    "get_module_version",
    "initialize_interface",
    "predict_genome",
    "predict_16S",
    "generate_kmer",
    "finalize_interface",
    "MODULE_VERSION",
]

# constants here
MODULE_VERSION = "1.0.0"

def get_current_path():
    return os.path.dirname(__file__)

def download(url, out):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(out, 'wb') as file, tqdm(
            desc='Downloading',
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# path settings
CURRENT_DIR = get_current_path()

# log files for python engine
LOG_DIR = "Xlassify_logs"
LOG_FILE = "Xlassify.log"
Logger = None
global model
model = None

DEBUG_CONSOLE_ON = False

def get_module_version():
    global MODULE_VERSION
    return MODULE_VERSION

def initialize_interface(flag, device_ids="0", log_level=0):
    global model
    global Logger
    global handle

    Logger = logging.getLogger("Xlassify")
    working_dir = os.getcwd()
    working_dir = os.path.join(working_dir, LOG_DIR)
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    log_file = os.path.join(working_dir, LOG_FILE)
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s:%(filename)s:%(lineno)s - %(message)s')
    handler.setFormatter(formatter)
    Logger.addHandler(handler)

    if DEBUG_CONSOLE_ON:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        Logger.addHandler(console)

    if log_level == 0:
        Logger.setLevel(logging.DEBUG)
    elif log_level == 1:
        Logger.setLevel(logging.WARN)
    elif log_level == 2:
        Logger.setLevel(logging.ERROR)
    elif log_level == 3:
        Logger.setLevel(logging.CRITICAL)
    elif log_level == 4:
        Logger.setLevel(logging.CRITICAL)

    # torch.cuda.set_device(int(device_ids))
    
    try:
        if model is None:
            Logger.info('Loading models...')
            # torch.backends.cudnn.deterministic = True
            if flag in ['species_genome', 'compute_kmer']:
                param_dict = {
                    'dropout':0.3,
                    'h_dim':256,
                    'n_label':1876,
                    'ft_dim':4**7,
                }   
                model = ResidualModel(**param_dict)
            else:
                if flag == 'species_full': n_targets = 193
                if flag == 'genus_full': n_targets = 156
                if flag == 'species_partial': n_targets = 407
                if flag == 'genus_partial': n_targets = 245
                model = RNA_Model(
                    use_model = 'CNN+MLP',
                    seq_mat_dim = 5,
                    seq_mer_dim = 4**7,
                    n_targets = n_targets,
                    hidden_size = 1024,
                    dropout = 0.7,
                )
            model = _load_model(flag, model)
            torch.cuda.empty_cache()
           # model.half()
            model.eval()
            Logger.info('Loading models successfully')

    except Exception as ex:
        Logger.error('Loading models failed')
        Logger.error(ex)
        return None
    return 1

def _get_kmer_mat_from_npy(kmer_file_lst, kmer_path):
    genome_name_lst = []
    kmer_ft_lst = []
    for kmer_file in kmer_file_lst:
        genome_name = os.path.splitext(kmer_file)[0]
        kmer_ft = np.load(osp.join(kmer_path, kmer_file))
        kmer_ft_lst.append(kmer_ft)
        genome_name_lst.append(genome_name)
    batch_kmer_ft = np.array(kmer_ft_lst)
    return batch_kmer_ft, genome_name_lst

def _main_genome(input_path, input_file_lst, save_path, save_file = 'res.csv', save_kmer_ft=True, nproc = 1):
    global Logger
    global model
    if input_file_lst[0].endswith('.npy'):
        kmer_utils._check_save_dir(save_path)
        batch_kmer_ft, genome_name_lst = _get_kmer_mat_from_npy(input_file_lst, input_path)
    else: # fasta
        fasta_path = input_path
        fasta_file_list = input_file_lst
        batch_kmer_ft, _, genome_name_lst = kmer_utils._preprocess_from_fasta(fasta_file_list, fasta_path, save_path, 
                                                                              save_kmer_ft, False, False, nproc)
    
    torch_kmer_ft = torch.tensor(batch_kmer_ft, dtype = torch.float32)
    pred = model(torch_kmer_ft)
    pred = torch.max(pred, 1)[1].numpy()
    res_df = pd.DataFrame(genome_name_lst, columns = ['Genome'])
    res_df['Pred_Label'] = pred
    label_info_file = osp.join(CURRENT_DIR,'dataset', 'species_label_df.csv')
    label_info_df = pd.read_csv(label_info_file)
    label_info_df = label_info_df[['Label','MGnify_accession','Lineage']].copy()
    label_info_df.columns = ['Pred_Label','Pred_MGnify', 'Pred_Lineage']		
    res_df = pd.merge(res_df, label_info_df, on = 'Pred_Label', how = 'left')
    res_df = res_df[['Genome', 'Pred_MGnify', 'Pred_Lineage']].copy()
    res_df.to_csv(osp.join(save_path, save_file), index=None)
    print("The results have saved in {}".format(osp.join(save_path, save_file)))
    return pred

def predict_genome(input_path, input_file_lst, save_path, save_file = 'results.csv', save_kmer_ft = True, nproc = 1):
    global Logger
    global handle
    code = "waited"
    Logger.info('start processing...')
    res=""
    st = time.time()
    res = _main_genome(input_path, input_file_lst, save_path, save_file, save_kmer_ft, nproc)
    torch.cuda.empty_cache()
    Logger.info('Cost time: {:.4f}'.format(time.time()-st))
    code='success'
    Logger.info('Done.')
    return res, code

def _main_16S(flag, input_path, input_file_lst, save_path, save_file = 'res.csv', save_kmer_ft=True, nproc = 1):
    global Logger
    global model
    if input_file_lst[0].endswith('.csv'): 
        batch_kmer_ft, batch_seq_ft, seq_name_lst = kmer_utils._preprocess_from_csv(
            input_file_lst[0], input_path, save_path, save_kmer_ft, get_seq_ft=True, save_pre_seq=False, nproc=nproc)
    else: # fasta
        batch_kmer_ft, batch_seq_ft, seq_name_lst = kmer_utils._preprocess_from_fasta(
            input_file_lst, input_path, save_path, save_kmer_ft, get_seq_ft=True, save_pre_seq=False, nproc=nproc)
    batch_seq_ft = np.array(batch_seq_ft, dtype = np.float32)
    torch_kmer_ft = torch.tensor(batch_kmer_ft, dtype = torch.float32)
    torch_seq_ft = torch.tensor(batch_seq_ft, dtype = torch.float32)

    pred = model(torch_seq_ft, torch_kmer_ft)
    pred = torch.max(pred, 1)[1].numpy()
    res_df = pd.DataFrame(seq_name_lst, columns = ['id'])
    res_df['Pred_Label'] = pred
    label_info_file = osp.join(CURRENT_DIR, 'dataset', flag+'.csv')
    label_info_df = pd.read_csv(label_info_file)
    if 'genus' in flag:
        label_info_df = label_info_df[['Label','Label_Name']]
    elif 'species' in flag:
        label_info_df = label_info_df[['Label','Label_Name','GTDB_lineage']]
    label_info_df = label_info_df.rename(columns = {'Label':'Pred_Label', 'Label_Name':'Pred_Name'})
    res_df = pd.merge(res_df, label_info_df, on = 'Pred_Label', how = 'left')
    if 'genus' in flag:
        res_df = res_df[['id', 'Pred_Name']]
        res_df.columns = ['id', 'Pred_GTDB']
    elif 'species' in flag:
        res_df = res_df[['id', 'Pred_Name','GTDB_lineage']]
        res_df.columns = ['id', 'Pred_MGnify', 'Pred_GTDB']
    res_df.to_csv(osp.join(save_path, save_file), index=None)
    return pred

def predict_16S(flag, input_path, input_file_lst, save_path, save_file = 'res.csv', save_kmer_ft=True, nproc = 1):
    global Logger
    global handle
    code = "waited"
    str_encode = None
    Logger.info('start processing...')
    res=""
    st = time.time()
    res = _main_16S(flag, input_path, input_file_lst, save_path, save_file, save_kmer_ft, nproc)
    torch.cuda.empty_cache()
    Logger.info('Cost time: {:.4f}'.format(time.time()-st))
    code='success'
    Logger.info('Done.')
    return res, code


def generate_kmer(input_path, input_file_lst, save_path, save_kmer_ft=True, nproc = 1):
    global Logger
    code = "waited"
    st = time.time()
    if input_file_lst[0].endswith('.npy'):
        kmer_utils._check_save_dir(save_path)
        print(input_file_lst)
        batch_kmer_ft, _ = _get_kmer_mat_from_npy(input_file_lst, input_path)
    if input_file_lst[0].endswith('.fasta'):
        batch_kmer_ft, _, _ = kmer_utils._preprocess_from_fasta(
            input_file_lst, input_path, save_path, save_kmer_ft, False, False, nproc)
    if input_file_lst[0].endswith('.csv'): 
        batch_kmer_ft, _, _ = kmer_utils._preprocess_from_csv(
            input_file_lst[0], input_path, save_path, save_kmer_ft, False, False, nproc)
    torch.cuda.empty_cache()
    Logger.info('Cost time: {:.4f}'.format(time.time()-st))
    code='success'
    Logger.info('Done.')
    return code

    
def finalize_interface():
    # finalize the interface
    global Logger
    h = Logger.handlers[0]
    try:
        h.acquire()
        h.flush()
        h.close()
    except (IOError, ValueError):
        pass
    finally:
        h.release()
    return 1

def _load_model(flag, model):
    print(f"load model params......")
    if flag in ['species_genome', 'compute_kmer']:
        model_file = 'ge5_le50_f5_m7_res_0_drop_seed0_f4_last.ckpt'
    if flag == 'genus_full':
        model_file = 'rg_mer=7_CNN+MLP_fold=10_cut=10_full_t8_100_seed=42_f2.ckpt'
    if flag == 'species_full':
        model_file = 'rs_mer=7_CNN+MLP_fold=10_cut=10_full_t8_100_seed=2077_f7.ckpt'
    if flag == 'genus_partial':
        model_file = 'rg_mer=7_CNN+MLP_fold=10_cut=10_partial_t8_100_seed=2077_f9.ckpt'
    if flag == 'species_partial':
        model_file = 'rs_mer=7_CNN+MLP_fold=10_cut=10_partial_t8_100_seed=2077_f3.ckpt'

    model_path = osp.join(CURRENT_DIR, 'model', model_file)
    if not osp.exists(model_path):
        print('model file does not exist!')
        model_url = 'https://github.com/SenseTime-Knowledge-Mining/Xlassify/raw/main/src/xlassify/model/' + model_file
        print(f'download model file from: {model_url}')
        download(model_url, model_path)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    print("load model params successful")
    return model

