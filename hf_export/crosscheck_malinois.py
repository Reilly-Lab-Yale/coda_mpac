"""Cross-check the exported Malinois against an independent prior computation.

Bitwise parity (asserted in convert.py) proves the *weights* survived conversion.
It says nothing about the surrounding usage: flank construction, reverse-complement
averaging, and one-hot encoding. Those live in this repo's `predict`, not in the
state dict.

This scores the exported Malinois on the chr7+chr13 test set and compares against
`alphapilot/phase1/results/p1_1a_malinois_published.tsv`, which was produced by a
separate loader (`foundation/src/boda_model.py`) from the same checkpoint. Agreement
validates the whole inference path end to end.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

sys.path.insert(0, '/nfs/roberts/project/pi_skr2/mcn26/coda_mpac/hf_export')
from malinois_hf.modeling_malinois import CELL_TYPES, MalinoisModel

EXPORT = '/nfs/roberts/scratch/pi_skr2/mcn26/mpac_hf/export'
TABLE = '/nfs/roberts/scratch/pi_skr2/go274/coda_data/DATA-Table_S2__MPRA_dataset.txt'
REFERENCE = '/nfs/roberts/project/pi_skr2/mcn26/alphapilot/phase1/results/p1_1a_malinois_published.tsv'
TEST_CHROMS = {'7', '13'}
TOLERANCE = 0.001


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = json.load(open(os.path.join(EXPORT, 'config.json')))

    model = MalinoisModel(**config)
    model.load_state_dict(load_file(os.path.join(EXPORT, 'model.safetensors')))
    model = model.eval().to(device)

    mpra = pd.read_table(TABLE, low_memory=False)
    mpra['chr'] = mpra['chr'].astype(str)
    mpra = mpra.loc[mpra['chr'].isin(TEST_CHROMS)]
    mpra = mpra.loc[mpra['sequence'].str.len() == 200].reset_index(drop=True)
    print(f'{mpra.shape[0]} sequences on chr{"+chr".join(sorted(TEST_CHROMS))}', flush=True)

    preds = model.predict(mpra['sequence'].tolist(), batch_size=256, device=device)
    for i, cell in enumerate(CELL_TYPES):
        mpra[f'{cell}_pred'] = preds[:, i].numpy()

    reference = pd.read_table(REFERENCE)
    se_cols = [f'{c}_lfcSE' for c in CELL_TYPES]

    failures = []
    for filter_name, cutoff in [('unfiltered', None), ('lfcSE<1.0', 1.0), ('lfcSE<0.5', 0.5)]:
        subset = mpra if cutoff is None else mpra.loc[mpra[se_cols].max(axis=1) < cutoff]
        print(f'\n{filter_name} (row mode), n = {subset.shape[0]}')
        for cell in CELL_TYPES:
            r = np.corrcoef(subset[f'{cell}_log2FC'], subset[f'{cell}_pred'])[0, 1]
            rho = subset[f'{cell}_log2FC'].corr(subset[f'{cell}_pred'], method='spearman')

            row = reference[(reference.filter_mode == 'row')
                            & (reference['filter'] == filter_name)
                            & (reference.cell == cell)]
            assert len(row) == 1, f"no unique reference row for {filter_name}/{cell}"
            want_r, want_n = float(row.pearson.iloc[0]), int(row.n.iloc[0])

            delta = abs(r - want_r)
            flag = 'OK ' if delta < TOLERANCE else 'DIFF'
            print(f'  {flag} {cell:6s} pearson {r:.4f} vs {want_r:.4f} (d={delta:.5f}), '
                  f'spearman {rho:.4f}, n {subset.shape[0]} vs {want_n}')
            if delta >= TOLERANCE or subset.shape[0] != want_n:
                failures.append(f'{filter_name}/{cell}')

    print()
    if failures:
        print(f'MISMATCH in {len(failures)} comparison(s): {failures}')
        sys.exit(1)
    print('all comparisons agree with the independent implementation')


if __name__ == '__main__':
    main()
