__version__ = '1.0.0'
__all__ = [
    '_get_seq_info',
    '_check_kmer_dir',
    '_check_preprocessed_sequence_dir',
    '_check_save_dir',
    '_check_dir',
    '_get_a_seq',
    'kmer_processor',
    '_get_a_seq_mat',
    '_preprocess_fasta',
    '_preprocess_from_fasta',
    '_preprocess_csv',
    '_preprocess_from_csv',]

import os
import time
import logging
import argparse
import numpy as np
from numpy.lib.index_tricks import IndexExpression
import pandas as pd
import os.path as osp
import multiprocessing as mp

parser = argparse.ArgumentParser(description='')
parser.add_argument('-k', type=int, default=7, help='The k of kmer.')
parser.add_argument('-i', '--input_path', type=str, help='The path of input file(.fasta).')
parser.add_argument('-f', '--input_file_lst', nargs='+', default='', help='The list of input file.(.fasta)')
parser.add_argument('-s', '--save_path', type=str, default='./results', help='The path of save file.')
class kmer_featurization:
    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'T', 'C', 'G']
        self.multiplyBy = 4 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
        self.n = 4**k # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

        Args:
            write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.    
        """
        kmer_features = []
        i = 0
        for seq in seqs:
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature)
            i+=1
            if i%1000==0:
                print(i)
        return np.asarray(kmer_features, dtype=np.float32)

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.

        Args:
            seq: 
            a string, a DNA sequence
            write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n)

        for i in range(number_of_kmers):
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer):
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering

def _get_seq_info():
    bases = "NTCGA"
    max_seq_len = 1748
    tcga_dict = dict(zip(bases, np.eye(len(bases))))
    return bases, max_seq_len, tcga_dict

# ========== check and make directory 
def _check_kmer_dir(kmer_path):
    if not osp.exists(kmer_path):
        print("Make a pre-processed kmer directory:", kmer_path)
        # logging.info("Make a pre-processed kmer directory: %s"%kmer_path)
        os.mkdir(kmer_path)
    else:
        print("The pre-processed kmer directory already exists.")
        # logging.info("The pre-processed kmer directory already exists.")

def _check_preprocessed_sequence_dir(seq_path):
    if not osp.exists(seq_path):
        print("Make a pre-processed sequence directory:", seq_path)
        os.mkdir(seq_path)
    else:
        print("The pre-processed sequence directory already exists.")

def _check_save_dir(save_path):
    if not osp.exists(save_path):
        print("Make a save directory:", save_path)
        os.mkdir(save_path)
    else:
        print("The save directory already exists.")

def _check_dir(save_path, save_kmer_ft=True, save_pre_seq=False):
    _check_save_dir(save_path)
    seq_path = osp.join(save_path, 'genome_pre')  
    kmer_path = osp.join(save_path, 'kmer')
    if save_kmer_ft:
        _check_kmer_dir(kmer_path)
    if save_pre_seq:
        _check_preprocessed_sequence_dir(seq_path)
    return seq_path, kmer_path

def _get_a_seq(fasta_file, fasta_path, seq_name=None, seq_path=None, save_pre_seq=False):
    with open(osp.join(fasta_path, fasta_file), 'r') as fr:
        a_seq = ''
        for line in fr:
            if not line.startswith('>'):
                a_seq += line.strip()
    pre_seq = a_seq.replace('N','')
   
    if save_pre_seq:
        save_seq_file = osp.join(seq_path, seq_name)
        with open(save_seq_file, 'w') as fw:
            fw.write(pre_seq)
    return a_seq, pre_seq

class kmer_processor:
    def __init__(self, k):
        self.k = k
        
    def get_a_kmer_file(self, kmer_path, seq_name):
        return osp.join(kmer_path, seq_name+'.npy')
        
    def check_a_kmer(self, kmer_file):
        return True if osp.exists(kmer_file) else False
        
    def get_a_kmer(self, pre_seq, kmer_file, save_kmer_ft=True):
        kmer_obj = kmer_featurization(k = self.k)
        kmer_ft = kmer_obj.obtain_kmer_feature_for_one_sequence(pre_seq).astype('float32')

        if save_kmer_ft:
            np.save(kmer_file, kmer_ft)
        return kmer_ft

def _get_a_seq_mat(a_seq, tcga_dict, max_seq_len = 1748, bases = "NTCGA"):
    seq_mat = ([tcga_dict[base] for base in a_seq])
    seq_mat = np.array(seq_mat)[:max_seq_len, :]
    if len(a_seq) < max_seq_len:
        residual = np.ones((max_seq_len - len(a_seq), len(bases)))*0.2
        seq_mat = np.concatenate((seq_mat, residual))
    return seq_mat

def _preprocess_fasta(fasta_file, fasta_path, seq_path, kmer_path, save_kmer_ft, get_seq_ft, save_pre_seq):
    seq_name = os.path.splitext(fasta_file)[0]
    seq_ft = []
    st=time.time()
    
    kmer_pro = kmer_processor(7)
    kmer_file = kmer_pro.get_a_kmer_file(kmer_path, seq_name)
    if kmer_pro.check_a_kmer(kmer_file) and not get_seq_ft:
        kmer_ft = np.load(kmer_file)
    else:
        a_seq, pre_seq = _get_a_seq(fasta_file, fasta_path,
            seq_name, seq_path, save_pre_seq)
        kmer_ft = kmer_pro.get_a_kmer(pre_seq, kmer_file, save_kmer_ft)
        if get_seq_ft:
            bases, max_seq_len, tcga_dict = _get_seq_info()
            seq_ft =_get_a_seq_mat(a_seq, tcga_dict, max_seq_len, bases)
            
    end=time.time()
    print("Preprocessing {}. Cost time {:.2f}".format(seq_name, end-st))
    return kmer_ft, seq_ft, seq_name

def _preprocess_from_fasta(fasta_file_list, fasta_path, save_path, save_kmer_ft=True, get_seq_ft=False, save_pre_seq=False, nproc = 1): # nproc=8, 19s
    seq_path, kmer_path = _check_dir(save_path, save_kmer_ft, save_pre_seq)
    seq_name_lst = []
    kmer_ft_lst = []
    seq_ft_lst = []
    with mp.Pool(nproc) as pool:
        multi_res_apply = [pool.apply_async(func=_preprocess_fasta, args = (fasta_file, fasta_path, seq_path, kmer_path, save_kmer_ft, get_seq_ft, save_pre_seq,)) 
                           for fasta_file in fasta_file_list if fasta_file.endswith('.fasta')]
        for res in multi_res_apply:
            kmer_ft, seq_ft, seq_name = res.get()
            kmer_ft_lst.append(kmer_ft)
            seq_ft_lst.append(seq_ft)
            seq_name_lst.append(seq_name)
    
    batch_kmer_ft = np.array(kmer_ft_lst)
    batch_seq_ft = np.array(seq_ft_lst)
    print("Pre-processing finished.")
    return batch_kmer_ft, batch_seq_ft, seq_name_lst

def _preprocess_csv(seq_name, pre_seq, a_seq, kmer_path, save_kmer_ft, get_seq_ft):
    seq_ft = []
    st=time.time()
    
    kmer_pro = kmer_processor(7)
    kmer_file = kmer_pro.get_a_kmer_file(kmer_path, seq_name)
    if kmer_pro.check_a_kmer(kmer_file) and not get_seq_ft:
        kmer_ft = np.load(kmer_file)
    else:
        kmer_ft = kmer_pro.get_a_kmer(pre_seq, kmer_file, save_kmer_ft)
        if get_seq_ft:
            bases, max_seq_len, tcga_dict = _get_seq_info()
            seq_ft =_get_a_seq_mat(a_seq, tcga_dict, max_seq_len, bases)
            
    end=time.time()
    print("Preprocessing {}. Cost time {:.2f}".format(seq_name, end-st))
    return kmer_ft, seq_ft, seq_name

def _preprocess_from_csv(csv_file, csv_path, save_path, save_kmer_ft=True, get_seq_ft=False, save_pre_seq=False, nproc = 1):
    _, kmer_path = _check_dir(save_path, save_kmer_ft, save_pre_seq)
    seq_name_lst = []
    kmer_ft_lst = []
    seq_ft_lst = []
    item_df = pd.read_csv(osp.join(csv_path, csv_file))
    item_df.columns = item_df.columns.str.lower()
    assert {'id', 'sequence'}.issubset(item_df.columns)
    item_df['sequence_kmer'] = item_df['sequence'].str.replace('N', '')
    if save_pre_seq:
        item_df.to_csv(osp.join(csv_path, 'pre'+csv_file), index=None)
    
    with mp.Pool(nproc) as pool:
        multi_res_apply = [pool.apply_async(func=_preprocess_csv, args = (seq_name, pre_seq, a_seq, kmer_path, save_kmer_ft, get_seq_ft,))
                           for seq_name, pre_seq, a_seq in zip(item_df['id'], item_df['sequence_kmer'], item_df['sequence'])]
        for res in multi_res_apply:
            kmer_ft, seq_ft, seq_name = res.get()
            kmer_ft_lst.append(kmer_ft)
            seq_ft_lst.append(seq_ft)
            seq_name_lst.append(seq_name)
    batch_kmer_ft = np.array(kmer_ft_lst)
    batch_seq_ft = np.array(seq_ft_lst)    
    return batch_kmer_ft, batch_seq_ft, seq_name_lst

if __name__ == '__main__':
    args = parser.parse_args() 
    k, input_path, save_path = args.k, args.input_path, args.save_path
    if args.input_file_lst == '':
        input_file_lst = os.listdir(input_path)
    elif len(args.input_file_lst) > 0:
        input_file_lst = args.input_file_lst
    
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)
    
    # 16S
    if input_path.endswith('fasta_16S'):
        batch_kmer_ft, batch_seq_ft, seq_name_lst = _preprocess_from_fasta(input_file_lst, input_path, save_path,
                                                             get_seq_ft=True, save_pre_seq=False)
        print(batch_kmer_ft.shape, batch_seq_ft.shape)
        
    elif input_path.endswith('csv_16S'):
        batch_kmer_ft, batch_seq_ft, seq_name_lst = _preprocess_from_csv(input_file_lst[0], input_path, save_path,
                                                             get_seq_ft=True, save_pre_seq=False)
        print(batch_kmer_ft.shape, batch_seq_ft.shape) 
    # genome
    elif input_path.endswith('fasta_genome'): 
        batch_kmer_ft, _, seq_name_lst = _preprocess_from_fasta(input_file_lst, input_path, save_path, 
                                                  get_seq_ft=False, save_pre_seq=True)
        print(batch_kmer_ft.shape)
			
    else: 
        batch_kmer_ft, _, seq_name_lst = _preprocess_from_fasta(input_file_lst, input_path, save_path, 
                                                  get_seq_ft=False, save_pre_seq=False, nproc = 1)
        # batch_kmer_ft, _, seq_name_lst = _preprocess_from_fasta_old(input_file_lst, input_path, save_path, 
        #                                           get_seq_ft=False, save_pre_seq=False)
        print(batch_kmer_ft.shape)

