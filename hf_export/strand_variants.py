"""Score Table S2 under all three strand-averaging schemes in one pass.

Settles whether `vcf_predict.py --average_full_revcomp TRUE` can reproduce the
preprint's Fig. 1B correlations. Each 200 bp insert is scored in four orientations:

    1  add_flanks(x)                      forward insert, forward flanks
    2  add_flanks(rc(x))                  RC insert, forward flanks   (chimeric)
    3  add_flanks(x).flip                 full reverse complement of 1
    4  add_flanks(rc(x)).flip             full reverse complement of 2 (chimeric)

and combined three ways:

    forward only  = 1                 no strand averaging at all
    convention B  = mean(1, 2)        vcf_predict.py default, what MPAC published
    convention A  = mean(1, 3)        training-time augmentation, tutorial, Fig. 1B
    four-way      = mean(1, 2, 3, 4)  vcf_predict.py --average_full_revcomp TRUE

Orientations 2 and 4 pair a flipped insert with unflipped vector context (or the
reverse), which corresponds to no real construct, so the four-way average is not a
clean switch between conventions -- it blends both with two artificial inputs.

Usage:
    python strand_variants.py --export <dir> --table_s2 <file> [--device cuda]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpac_hf.modeling_mpac import CELL_TYPES, dna2tensor
from verify import build_fold_index, load_fold_models

VARIABLE_REGION_LEN = 200
SCHEMES = ['forward_only', 'convention_B', 'convention_A', 'four_way']


@torch.no_grad()
def score_all_schemes(ensemble, sequences, device, batch_size=128):
    """Return {scheme: (n, 3) tensor} for one fold's ensemble."""
    out = {k: [] for k in SCHEMES}
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        x = torch.stack([dna2tensor(s.upper()) for s in chunk]).to(device)

        fwd = ensemble.add_flanks(x)                      # 1
        chi = ensemble.add_flanks(x.flip(dims=[1, 2]))    # 2

        p1 = ensemble(fwd)
        p2 = ensemble(chi)
        p3 = ensemble(fwd.flip(dims=[1, 2]))              # 3
        p4 = ensemble(chi.flip(dims=[1, 2]))              # 4

        out['forward_only'].append(p1.cpu())
        out['convention_B'].append(((p1 + p2) / 2).cpu())
        out['convention_A'].append(((p1 + p3) / 2).cpu())
        out['four_way'].append(((p1 + p2 + p3 + p4) / 4).cpu())
    return {k: torch.cat(v, dim=0) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', required=True)
    parser.add_argument('--table_s2', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = json.load(open(os.path.join(args.export, 'config.json')))
    fold_index = build_fold_index(args.export)

    mpra = pd.read_table(args.table_s2, low_memory=False)
    mpra = mpra.loc[mpra['sequence'].str.len() == VARIABLE_REGION_LEN]
    mpra['chr'] = mpra['chr'].astype(str)
    mpra = mpra.loc[mpra['chr'].isin(fold_index)].reset_index(drop=True)
    print(f'{mpra.shape[0]} sequences on covered chromosomes', flush=True)

    frames = []
    for chrom, group in mpra.groupby('chr', sort=True):
        ensemble, n_models = load_fold_models(args.export, config, fold_index[chrom], args.device)
        preds = score_all_schemes(ensemble, group['sequence'].tolist(),
                                  args.device, args.batch_size)
        frame = group[[f'{c}_log2FC' for c in CELL_TYPES]].copy()
        for scheme, tensor in preds.items():
            assert tensor.shape == (group.shape[0], len(CELL_TYPES)), \
                f"{scheme}: got {tuple(tensor.shape)} for {group.shape[0]} sequences"
            for i, cell in enumerate(CELL_TYPES):
                frame[f'{cell}__{scheme}'] = tensor[:, i].numpy()
        frames.append(frame)
        print(f'chr{chrom}: {group.shape[0]} seqs, {n_models} models', flush=True)
        del ensemble
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    results = pd.concat(frames, ignore_index=True)

    print(f'\nPearson r vs empirical MPRA activity, n = {results.shape[0]} (no lfcSE filter)')
    print(f'{"scheme":16s} ' + ' '.join(f'{c:>9s}' for c in CELL_TYPES))
    for scheme in SCHEMES:
        rs = [np.corrcoef(results[f'{c}_log2FC'], results[f'{c}__{scheme}'])[0, 1]
              for c in CELL_TYPES]
        print(f'{scheme:16s} ' + ' '.join(f'{r:9.4f}' for r in rs))
    print(f'{"preprint Fig 1B":16s} ' + ' '.join(f'{v:9.2f}' for v in [0.89, 0.89, 0.88]))

    out_path = os.path.join(args.export, 'strand_variants.tsv.gz')
    results.to_csv(out_path, sep='\t', index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
