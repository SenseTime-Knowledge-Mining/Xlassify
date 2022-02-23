# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import argparse
import base64
import os.path as osp
import xlassify.interface as interface
from xlassify.interface import predict_16S, predict_genome, generate_kmer
from xlassify.interface import finalize_interface, initialize_interface, get_current_path
'''
python interface_test.py -m species_genome -r res_genome1.csv --nproc 8 --save_kmer 0 

python interface_test.py -m species_genome -i ../fasta_genome -r res_genome1.csv --nproc 8 --save_kmer 0 
python interface_test.py -m species_genome -i ../fasta_genome -r res_genome2.csv --nproc 8 
python interface_test.py -m species_genome -i ../fasta_genome -r res_genome --nproc 8 -b 4

python interface_test.py -m species_genome -i ../kmer_genome -r res_kmer.csv --nproc 8
python interface_test.py -m species_genome -i ../kmer_genome -r res_kmer --nproc 8 -b 4

python interface_test.py -m compute_kmer -i ../fasta_genome -s kmer7_genome -k 7 --nproc 8

python interface_test.py -m genus_full -i ../fasta_16S -s results_16S -r res_16S_genus.csv --nproc 8 --save_kmer 0 
python interface_test.py -m species_full -i ../fasta_16S -s results_16S -r res_16S_species.csv --nproc 8 
python interface_test.py -m genus_full -i ../fasta_16S -s results_16S -r res_16S_genus --nproc 8 -b 4

python interface_test.py -m species_full -i ../csv_16S -f sf_test.csv -s results_16S -r sf_res.csv --nproc 8
python interface_test.py -m genus_full -i ../csv_16S -f gf_test.csv -s results_16S -r gf_res.csv --nproc 8 -b 4 # batch不能用

python interface_test.py -m compute_kmer -i ../fasta_16S -s kmer7_16S -k 7 --nproc 8
'''
def read_params():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model_name', type=str, default='species_genome', help=
                        'Choose a model from {compute_kmer, species_genome, genus_full, species_full}. Default: species_genome')
    parser.add_argument('-i', '--input_path', type=str, default=get_current_path()+'/dataset/test', help='The path of input fasta file. Using testing data as default.')
    parser.add_argument('-f', '--input_file_lst', nargs='+', default='', help='The list of input file.')
    parser.add_argument('-s', '--save_path', type=str, default='./Xlassify_results', help='The path of save file. Default: ./Xlassify_results')
    parser.add_argument('-r', '--save_file', type=str, default='res.csv', help='The path of results file. Default: res.csv')
    parser.add_argument('--save_kmer', type=int, default=1, help='Save kmer or not {0,1}. Default: 1')
    parser.add_argument('-b', '--batch', type=int, default=0, help='The batch of prediction.')
    parser.add_argument('-k', type=int, default=7, help='The k of kmer. Default: 7')
    parser.add_argument('--nproc', type=int, default=1, help='The number of CPUs to use. Default: 1')
    return parser.parse_args()

def main():
    interface.DEBUG_CONSOLE_ON = True
    args = read_params()
    input_path, save_path, save_file = args.input_path, args.save_path, args.save_file
    nproc, model_name, k, save_kmer_ft = args.nproc, args.model_name, args.k, args.save_kmer
    res = ""
    initialize_interface(model_name, device_ids="0")
    print("Processing...")
    st = time.time()
    if args.input_file_lst:
        input_file_lst = args.input_file_lst
    else:
        input_file_lst = os.listdir(input_path) 
    if bool(args.batch) and input_file_lst[0].endswith('.csv'): 
        print(".csv format don't support batch format.")
        args.batch = 0
    if bool(args.batch):

        batch_size = args.batch
        n = math.ceil(len(input_file_lst)/batch_size)
        for i in range(n):
            print("Batch {} begins...".format(i+1))
            batch_save_file = '%s_batch%d.csv' %(osp.basename(save_file),i+1)
            if model_name == 'species_genome':
                pred, _ = predict_genome(input_path, input_file_lst[i*batch_size:(i+1)*batch_size], save_path, batch_save_file, save_kmer_ft, nproc)
            elif model_name == 'compute_kmer':
                _ = generate_kmer(input_path, input_file_lst, save_path, k, nproc)
            else:
                pred, _ = predict_16S(model_name, input_path, input_file_lst[i*batch_size:(i+1)*batch_size], save_path, batch_save_file, nproc)
            
    else:
        if model_name == 'species_genome':
            pred, _ = predict_genome( input_path, input_file_lst, save_path, save_file, save_kmer_ft, nproc)
        elif model_name == 'compute_kmer':
            _ = generate_kmer(input_path, input_file_lst, save_path, k, nproc)
        else:
            pred, _ = predict_16S(model_name, input_path, input_file_lst, save_path, save_file, save_kmer_ft, nproc)
        
    end=time.time()  
    print("Total time is: {:.2f}".format(end-st))

if __name__ == "__main__":
    main()