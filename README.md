# Xlassify

Fast and accurate taxonomic classification of bacteria genomes is a key step in human gut microbiome analysis. Here we propose Xlassify, an alignment-free deep-learning model that is specifically trained to classify human gut bacteria.


### Architecture

16S model:
![16s_model](./docs/images/16s_model.png)

genome model:
![genome_model](./docs/images/genome_model.png)


### Descriptions  

The most important files in this projects are as follow:
```bash
.
├── 16s                         # Xlassify 16s model
│   ├── data                    # 16s dataset
│   ├── slurm_split_data        # processed 16s data
│   ├── kmer.py                 # create K-mer feature
│   ├── model_chg3.py           # 16s model
│   ├── model_chg_mnb.py        # naive Bayesian classifier model
│   ├── model_chg_rdp.py        # RDP classifier model
│   ├── model_chg_rf.py         # random forest model
│   ├── model.py                # implementation of 16s model
│   └── fold.sh                 # train scripts for 16s model and other baselines
├── genome                      # Xlassify genome model
│   ├── data                    # genome dataset
│   ├── datasetG7m.py           # create K-mer feature
│   ├── model.py                # implementation of genome model
│   ├── trainer7m.py            # genome model
│   └── ru.sh                   # train scripts
├── docs
└── README.md
```


### Installation

```bash
conda create -n py37 python=3.7.1
conda activate py37
pip install -r requirements.txt
```


### Usage

For 16S model, modify the parameters in `16s/fold.sh` file:
```
k=7                 # kmer size
t="t"               # logname marker
model="CNN+MLP"     # options are CNN, MLP, CNN+MLP, RF, and RDP
th=10               # k-fold cross-validation
data_type="full"    # full or partial
```

Then execute the following scripts to train and test 16S model:
```bash
cd 16s
chmod +777 fold.sh
./fold.sh
```

 For genome model, just execute the following scripts to train and test genome model:
```bash
cd genome
chmod +777 run.sh
./run.sh
```
