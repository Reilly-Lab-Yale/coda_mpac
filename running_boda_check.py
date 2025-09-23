import shutil
import os

import torch
import torch.nn as nn

import boda

# Load Models interactively...
# Get Malinois
malinois_path = 'gs://tewhey-public-data/CODA_resources/malinois_artifacts__20211113_021200__287348.tar.gz'
my_model = boda.common.utils.load_model(malinois_path)
input_len = torch.load('./artifacts/torch_checkpoint.pt')['model_hparams'].input_len
# Set Flanks
left_pad_len = (input_len - 200) // 2
right_pad_len= (input_len - 200) - left_pad_len

left_flank = boda.common.utils.dna2tensor( 
    boda.common.constants.MPRA_UPSTREAM[-left_pad_len:] 
).unsqueeze(0)
print(f'left flank shape: {left_flank.shape}')

right_flank= boda.common.utils.dna2tensor( 
    boda.common.constants.MPRA_DOWNSTREAM[:right_pad_len] 
).unsqueeze(0)
right_flank.shape
print(f'right flank shape: {right_flank.shape}')

flank_builder = boda.common.utils.FlankBuilder(
    left_flank=left_flank,
    right_flank=right_flank,
)

flank_builder.cuda()
# Example Call
#placeholder = torch.randn((10,4,200)).cuda() # Simulate a batch_size x 4 nucleotide x 200 nt long sequence
#prepped_seq = flank_builder( placeholder )   # Need to add MPRA flanks

#with torch.no_grad():
#    print( my_model( prepped_seq ) )

# Run on MPRA data set
import pandas as pd
import numpy as np
import csv
from scipy.stats import pearsonr, spearmanr
import tqdm as tqdm
import matplotlib.pyplot as plt
###
haplotypes = pd.read_table('for_john.txt', sep='\t', header=0)

pass_seq = haplotypes.loc[ haplotypes['oligo'].str.len() == 200 ].reset_index(drop=True)

seq_tensor  = torch.stack([ boda.common.utils.dna2tensor(x['oligo']) for i, x in tqdm.tqdm(pass_seq.iterrows(), total=pass_seq.shape[0]) ], dim=0)
seq_dataset = torch.utils.data.TensorDataset(seq_tensor)
seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=128)
###
results = []

with torch.no_grad():
    for i, batch in enumerate(tqdm.tqdm(seq_loader)):
        prepped_seq = flank_builder( batch[0].cuda() )
        predictions = my_model( prepped_seq ) + \
                      my_model( prepped_seq.flip(dims=[1,2]) ) # Also
        predictions = predictions.div(2.)
        results.append(predictions.detach().cpu())

predictions = torch.cat(results, dim=0)
###
pred_df     = pd.DataFrame( predictions.numpy(), columns=['K562_preds', 'HepG2_preds', 'SKNSH_preds'] )
all_results = pd.concat([pass_seq, pred_df], axis=1)
print(all_results.head())