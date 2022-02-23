import numpy as np
'''序列过短可能有nan

例子
k = 8  # choose the value for k
obj = kmer_featurization(k)  # initialize a kmer_featurization object
annotation["sequence"] = annotation["sequence"].str.replace("N","")
annotation.reset_index(drop=True,inplace=True)
kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(annotation["sequence"], write_number_of_occurrences=False)
'''
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
    # kmer_features = np.array(kmer_features)
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


'''
kmer_lst = reduce(lambda x,y: [i+j for i in x for j in y], [self.c_lst] * k)
self.kmer_dim = len(kmer_lst)
print(self.kmer_dim)
        
kmer_ft = np.zeros((len(seq_lst), len(kmer_lst)), dtype=np.float32)
for idx, seq in enumerate(seq_lst):
    for i in range(len(seq)-k+1):
        kmer_ft[idx,kmer_lst.index(seq[i:k+i])]+=1
    if normalization:
        kmer_ft[idx]/=(len(seq)-k+1)



kmer_ft = []*len(seq_lst)#np.zeros((len(seq_lst), len(kmer_lst)), dtype=np.float32)
for idx, seq in enumerate(seq_lst):
    kmer = [0]*self.kmer_dim
    for i in range(len(seq)-k+1):
        kmer[kmer_lst.index(seq[i:k+i])]+=1
    kmer_ft.append(kmer)
kmer_ft = np.array(kmer_ft, dtype=np.float32)
if normalization:
    kmer_ft/= (np.array(self.seq_len_lst, dtype=np.float32)-k+1)[:,None]
'''
if __name__ == '__main__':
    pass
