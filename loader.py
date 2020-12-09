#-*- coding:utf-8 -*- 

import os
import torch as t
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import numpy as np
np.random.seed(1080)

class SPOTDataSetDot(data.Dataset):
    def __init__(self, dataset_folder, device):
        'Initialization'
        super(SPOTDataSetDot, self).__init__()
        self.dataset_folder = dataset_folder
        self.device = device
        self.samples = self.load_sample_list()
    
    def load_sample_list(self):
        samples = []
        for line in open( os.path.join(self.dataset_folder,'filelist'), 'r' ):
            rna_name, seq_fn, dot_fn = line.strip().split()
            seq_fn = os.path.join(self.dataset_folder, seq_fn)
            dot_fn = os.path.join(self.dataset_folder, dot_fn)
            seq_matrix = t.FloatTensor( np.fromfile(seq_fn, sep=',').reshape(-1,4) )
            dot_matrix = t.FloatTensor( np.fromfile(dot_fn, sep=',').reshape(-1,1) )
            samples.append( (rna_name, seq_matrix, dot_matrix) )
        return samples
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)
    
    def seqmatrix_coding(self, seq_matrix):
        n = seq_matrix.shape[0]
        one_hot = seq_matrix.unsqueeze(0)
        one_hot = one_hot.repeat([n,1,2])
        one_hot[:,:,4] = one_hot[:,:,0].t()
        one_hot[:,:,5] = one_hot[:,:,1].t()
        one_hot[:,:,6] = one_hot[:,:,2].t()
        one_hot[:,:,7] = one_hot[:,:,3].t()
        one_hot = one_hot.permute([2,0,1])
        one_hot = one_hot.unsqueeze(0).contiguous()
        return one_hot
    
    def __getitem__(self, index):
        'Generates one sample of data'
        rna_name, seq_matrix, dot_matrix = self.samples[index]
        seq_matrix = seq_matrix.to(self.device)
        dot_matrix = dot_matrix.to(self.device)
        seq_matrix = self.seqmatrix_coding(seq_matrix)
        
        return (rna_name, seq_matrix), dot_matrix
