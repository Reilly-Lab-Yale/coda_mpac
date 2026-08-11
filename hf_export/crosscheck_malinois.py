"""Cross-check the ported model against an independent prior computation.

Bitwise parity (asserted in convert.py) proves the *weights* survived conversion.
It says nothing about the surrounding usage: flank construction, reverse-complement
averaging, and one-hot encoding. Those live in this repo's `predict`, not in the
state dict.

This scores the original Malinois checkpoint through the ported architecture on the
chr7+chr13 test set and compares against
`alphapilot/phase1/results/p1_1a_malinois_published.tsv`, which was produced by a
separate loader (`foundation/src/boda_model.py`) from the same checkpoint. Agreement
validates the whole inference path end to end.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import normalize_hparams
from mpac_hf.modeling_mpac import CELL_TYPES, MPACModel

# The original Malinois checkpoint, loaded directly rather than from the export:
# the MPAC-only export publishes no single all-chromosome model, and the reference
# numbers below were computed from this checkpoint.
CKPT = '/nfs/roberts/scratch/pi_skr2/go274/coda_data/artifacts/torch_checkpoint.pt'
TABLE = '/nfs/roberts/scratch/pi_skr2/go274/coda_data/DATA-Table_S2__MPRA_dataset.txt'
TEST_CHROMS = {'7', '13'}
TOLERANCE = 0.001

# Reference values from `alphapilot/phase1/results/p1_1a_malinois_published.tsv`,
# computed by a separate loader (`foundation/src/boda_model.py`) from the same
# checkpoint, row-mode lfcSE filtering.
#
# Transcribed rather than read from disk because that file was destroyed on
# 2026-08-11 along with the rest of alphapilot/ and foundation/. Before it was lost,
# this script reproduced every value below to five decimal places in both Pearson
# and Spearman (job 21967070), which is the evidence that the transcription is
# faithful. Restoring the originals from backup would let this read the file again.
#   filter -> cell -> (pearson, spearman, n)
REFERENCE = {
    'unfiltered': {
        'K562':  (0.8709646372164659, 0.7929344485749014, 63958),
        'HepG2': (0.8734377440231157, 0.8179699788955331, 63958),
        'SKNSH': (0.8605971153845648, 0.8124109150557576, 63958),
    },
    'lfcSE<1.0': {
        'K562':  (0.8842019717134781, 0.8103576392377410, 60055),
        'HepG2': (0.8880156271920135, 0.8333900952534008, 60055),
        'SKNSH': (0.8785368668971479, 0.8305707933083167, 60055),
    },
    'lfcSE<0.5': {
        'K562':  (0.8888014875455232, 0.8220269332295450, 54162),
        'HepG2': (0.8917344489074619, 0.8402410169322609, 54162),
        'SKNSH': (0.8849062024787797, 0.8404278688990058, 54162),
    },
}


@torch.no_grad()
def predict_construct_flip(model, sequences, device, batch_size=256):
    """Score sequences averaging the forward pass with a flip of the flanked tensor.

    This is NOT what `MPACModel.predict` does -- that follows `vcf_predict.py` and
    reverse-complements the insert alone. The reference numbers in
    `p1_1a_malinois_published.tsv` were produced with the whole-construct flip, so
    the comparison has to use it too.

    Keeping this separate is the point: it pins down the weights, the one-hot
    encoding, the flank construction and the forward pass against an outside
    implementation, without depending on which strand convention `predict` uses.
    """
    from mpac_hf.modeling_mpac import dna2tensor

    out = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.stack([dna2tensor(s.upper())
                             for s in sequences[start:start + batch_size]]).to(device)
        prepped = model.add_flanks(batch)
        preds = (model(prepped) + model(prepped.flip(dims=[1, 2]))).div(2.)
        out.append(preds.cpu())
    return torch.cat(out, dim=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False)
    config = normalize_hparams(checkpoint)
    model = MPACModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    mpra = pd.read_table(TABLE, low_memory=False)
    mpra['chr'] = mpra['chr'].astype(str)
    mpra = mpra.loc[mpra['chr'].isin(TEST_CHROMS)]
    mpra = mpra.loc[mpra['sequence'].str.len() == 200].reset_index(drop=True)
    print(f'{mpra.shape[0]} sequences on chr{"+chr".join(sorted(TEST_CHROMS))}', flush=True)

    preds = predict_construct_flip(model, mpra['sequence'].tolist(), device)
    for i, cell in enumerate(CELL_TYPES):
        mpra[f'{cell}_pred'] = preds[:, i].numpy()

    se_cols = [f'{c}_lfcSE' for c in CELL_TYPES]

    failures = []
    for filter_name, cutoff in [('unfiltered', None), ('lfcSE<1.0', 1.0), ('lfcSE<0.5', 0.5)]:
        subset = mpra if cutoff is None else mpra.loc[mpra[se_cols].max(axis=1) < cutoff]
        print(f'\n{filter_name} (row mode), n = {subset.shape[0]}')
        for cell in CELL_TYPES:
            r = np.corrcoef(subset[f'{cell}_log2FC'], subset[f'{cell}_pred'])[0, 1]
            rho = subset[f'{cell}_log2FC'].corr(subset[f'{cell}_pred'], method='spearman')

            want_r, want_rho, want_n = REFERENCE[filter_name][cell]

            d_r, d_rho = abs(r - want_r), abs(rho - want_rho)
            ok = d_r < TOLERANCE and d_rho < TOLERANCE and subset.shape[0] == want_n
            print(f'  {"OK " if ok else "DIFF"} {cell:6s} '
                  f'pearson {r:.4f} vs {want_r:.4f} (d={d_r:.5f}), '
                  f'spearman {rho:.4f} vs {want_rho:.4f} (d={d_rho:.5f}), '
                  f'n {subset.shape[0]} vs {want_n}')
            if not ok:
                failures.append(f'{filter_name}/{cell}')

    print()
    if failures:
        print(f'MISMATCH in {len(failures)} comparison(s): {failures}')
        sys.exit(1)
    print('all comparisons agree with the independent implementation')


if __name__ == '__main__':
    main()
