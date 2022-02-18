#!/bin/sh 
# Ends with t is temp
k=7
t="t"
model="CNN+MLP" # CNN+MLP
# model="RF"
th=10
data_type="full" # full partial

logname_s="rs_mer="$k"_"$model"_fold="$th"_cut="$th"_"$data_type"_"$t
logname_g="rg_mer="$k"_"$model"_fold="$th"_cut="$th"_"$data_type"_"$t

# My model: MLP, CNN, CNN+MLP; not use slurm_split_data;
nohup python -u model_chg3.py --log $logname_s -k $k --use_model $model --data_type $data_type --threshold $th -l species > res/$logname_s 2>&1 &
nohup python -u model_chg3.py --log $logname_g -k $k --use_model $model --data_type $data_type --threshold $th -l genus   > res/$logname_g 2>&1 &

# RF: use slurm_split_data;
# nohup python -u model_chg_rf.py --log $logname_s -k $k --data_type $data_type --threshold $th -l species > res/$logname_s 2>&1 &
# nohup python -u model_chg_rf.py --log $logname_g -k $k --data_type $data_type --threshold $th -l genus   > res/$logname_g 2>&1 &

# MNB: use slurm_split_data;
# nohup python -u model_chg_mnb.py --log $logname_s -k $k --data_type $data_type --threshold $th -l species > res/$logname_s 2>&1 &
# nohup python -u model_chg_mnb.py --log $logname_g -k $k --data_type $data_type --threshold $th -l genus   > res/$logname_g 2>&1 &

# RDP: use slurm_split_data; without prior P; 
# nohup python -u model_chg_rdp.py --log $logname_s -k $k --data_type $data_type --threshold $th -l species > res/$logname_s 2>&1 &
# nohup python -u model_chg_rdp.py --log $logname_g -k $k --data_type $data_type --threshold $th -l genus   > res/$logname_g 2>&1 &

################################################# old code #######################################################################
# ==================== YW code
# nohup python model_cnn.py full 10 16S species > logs/fus_species_full_16S_10_f10_1 2>&1 &
# nohup python model_cnn.py full 10 16S genus > logs/fus_genus_full_16S_10_f10_1 2>&1 &

# ==================== (unkonwn?)
# nohup python trainer_kmer_MLP_nfold.py genus full_16S 10 > kfold/genus_full_16S_10_p500 2>&1 &
# nohup python trainer_kmer_MLP_nfold.py genus full_V3V4 10 > kfold/genus_full_V3V4_10_o20 2>&1 &

# ==================== 5-fold, 5 cutoff or 10 cutoff
# nohup python trainer_kmer_MLP_kfold.py genus full_16S 5 > kfold/genus_full_16S_5_f5 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py genus full_V3V4 5 > kfold/genus_full_V3V4_5_f5 2>&1 &

# nohup python trainer_kmer_MLP_kfold.py species full_16S 5 > kfold/species_full_16S_5_f5 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py species full_V3V4 5 > kfold/species_full_V3V4_5_f5 2>&1 &

# nohup python trainer_kmer_MLP_kfold.py genus full_16S 10 > kfold/genus_full_16S_10_f5 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py genus full_V3V4 10 > kfold/genus_full_V3V4_10_f5 2>&1 &

# nohup python trainer_kmer_MLP_kfold.py species full_16S 10 > kfold/species_full_16S_10_f5 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py species full_V3V4 10 > kfold/species_full_V3V4_10_f5 2>&1 &

# ==================== 10-fold
# nohup python trainer_kmer_MLP_kfold.py genus full_16S 10 > kfold/genus_full_16S_10_f10 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py genus full_V3V4 10 > kfold/genus_full_V3V4_10_f10 2>&1 &

# nohup python trainer_kmer_MLP_kfold.py species full_16S 10 > kfold/species_full_16S_10_f10 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py species full_V3V4 10 > kfold/species_full_V3V4_10_f10 2>&1 &

# ==================== leave-one-out
# nohup python trainer_kmer_MLP_nfold.py genus full_V3V4 10 > kfold/genus_full_V3V4_10_loo_ 2>&1 &
# nohup python trainer_kmer_MLP_nfold.py species full_V3V4 10 > kfold/species_full_V3V4_10_loo_ 2>&1 &

# ==================== 10-fold by nfold.py
# nohup python trainer_kmer_MLP_nfold.py genus full_V3V4 10 > kfold/genus_full_V3V4_10_f10_ 2>&1 &
# nohup python trainer_kmer_MLP_nfold.py species full_V3V4 10 > kfold/species_full_V3V4_10_f10_ 2>&1 &

# ==================== 10 StratifiedKFold 
# nohup python trainer_kmer_MLP_kfold.py genus full_V3V4 10 > kfold/genus_full_V3V4_10_hf10_ 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py species full_V3V4 10 > kfold/species_full_V3V4_10_hf10_ 2>&1 &

# ==================== all in full
# nohup python trainer_kmer_MLP6_kfold.py genus all_16S 10 > all_full5 2>&1 &
# nohup python trainer_kmer_MLP6_kfold.py genus all_V3V4 10 > all_full6 2>&1 &

# nohup python trainer_kmer_MLP6_kfold.py species all_16S 10 > all_full7 2>&1 &
# nohup python trainer_kmer_MLP6_kfold.py species all_V3V4 10 > all_full8 2>&1 &

# ==================== all
# nohup python trainer_kmer_MLP_kfold.py genus all_16S 5 > kfold/genus_all_16S_5_f5_ 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py genus all_V3V4 5 > kfold/genus_all_V3V4_5_f5_ 2>&1 &

# nohup python trainer_kmer_MLP_kfold.py species all_16S 5 > kfold/species_all_16S_5_f5_ 2>&1 &
# nohup python trainer_kmer_MLP_kfold.py species all_V3V4 5 > kfold/species_all_V3V4_5_f5_ 2>&1 &

# ==================== all (old?)
# nohup python trainer_kmer_MLP6_kfold.py genus all_16S 5 > kfold/genus_all_16S_5_f5 2>&1 &
# nohup python trainer_kmer_MLP6_kfold.py genus all_V3V4 5 > kfold/genus_all_V3V4_5_f5 2>&1 &

# nohup python trainer_kmer_MLP6_kfold.py species all_16S 5 > kfold/species_all_16S_5_f5 2>&1 &
# nohup python trainer_kmer_MLP6_kfold.py species all_V3V4 5 > kfold/species_all_V3V4_5_f5 2>&1 &

# ==================== fusion kmer and seq
# nohup python trainer_fusion_kfold.py genus full_16S 10 > logs/fus_genus_full_16S_10 2>&1 &
# nohup python trainer_fusion_kfold.py species full_16S 10 > logs/fus_species_full_16S_10 2>&1 &

# ==================== 6mer
# nohup python trainer_kmer_MLP_kfold.py genus full_16S 10 > kfold/genus_full_16S_10_f5_mer6 2>&1 &


