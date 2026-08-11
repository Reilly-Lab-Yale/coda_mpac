"""
End-to-end check of the exported MPAC weights.

Reproduces the correlation between predicted and empirical MPRA activity
reported in the MPAC preprint (Fig. 1B). Each sequence is scored by the
ensemble of ten models whose *test* fold contains that sequence's source
chromosome, so no sequence is ever scored by a model that trained on it. This
is the property that defines MPAC, and it is the thing most likely to be
silently lost in a repackaging.

Usage:
    python verify.py --export <dir> --table_s2 <Table_S2__MPRA_dataset.txt> \
        [--sample 20000] [--device cuda]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpac_hf.modeling_mpac import CELL_TYPES, MPACEnsemble, MPACModel

VARIABLE_REGION_LEN = 200

# Every sequence is scored once; the activity-noise (lfcSE) filter is applied
# afterwards, since it is an analysis choice rather than a property of the model.
# The preprint reports a different n per cell type, which only a per-column
# filter can produce -- a row-wise filter drops the same rows for all three.
FILTERS = [
    ('none', None, 'row'),
    ('row-wise lfcSE < 1.0', 1.0, 'row'),
    ('per-cell lfcSE < 1.0', 1.0, 'col'),
    ('row-wise lfcSE < 0.5', 0.5, 'row'),
    ('per-cell lfcSE < 0.5', 0.5, 'col'),
]


def load_fold_models(export_dir, config, fold_dir, device):
    """Load the ten replicate models for one test fold into a vmap ensemble."""
    paths = sorted(
        os.path.join(fold_dir, f) for f in os.listdir(fold_dir) if f.endswith('.safetensors')
    )
    assert len(paths) > 0, f"no weights in {fold_dir}"

    models = []
    for path in paths:
        model = MPACModel(**config)
        model.load_state_dict(load_file(path))
        model.eval()
        models.append(model.to(device))
    return MPACEnsemble(models).to(device), len(paths)


def build_fold_index(export_dir):
    """Map source chromosome -> directory of models that held it out as test."""
    with open(os.path.join(export_dir, 'provenance.json')) as handle:
        records = json.load(handle)

    index = {}
    for record in records:
        if record.get('role') != 'crossval':
            continue
        fold_dir = os.path.dirname(os.path.join(export_dir, record['file']))
        for chrom in record['test_chrs']:
            existing = index.setdefault(chrom, fold_dir)
            assert existing == fold_dir, \
                f"chromosome {chrom} maps to two folds: {existing} and {fold_dir}"
    return index


def report(results, label, cutoff, mode):
    """Print per-cell-type Pearson r under one lfcSE filtering scheme.

    'row' drops a sequence entirely if any cell type's standard error exceeds the
    cutoff; 'col' drops only the offending cell type's measurement, so each cell
    type ends up with its own n.
    """
    print(f'\n{label}')
    stats = {}
    for cell in CELL_TYPES:
        obs = results[f'{cell}_log2FC']
        pred = results[f'{cell}_pred']
        keep = obs.notna() & pred.notna()

        if cutoff is not None:
            se_cols = [f'{c}_lfcSE' for c in CELL_TYPES]
            if mode == 'row':
                keep &= results[se_cols].max(axis=1) < cutoff
            else:
                keep &= results[f'{cell}_lfcSE'] < cutoff

        n = int(keep.sum())
        assert n > 0, f"{label}: no rows left for {cell}"
        r = np.corrcoef(obs[keep], pred[keep])[0, 1]
        stats[cell] = (r, n)
        print(f'  {cell:6s} Pearson r = {r:.4f}  (n = {n})')
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', required=True, help='Directory written by convert.py')
    parser.add_argument('--table_s2', required=True, help='Table_S2__MPRA_dataset.txt')
    parser.add_argument('--sample', type=int, default=None,
                        help='Randomly subsample this many sequences per fold (for quick runs)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.export, 'config.json')) as handle:
        config = json.load(handle)

    fold_index = build_fold_index(args.export)
    print(f'{len(fold_index)} chromosomes covered by crossval folds', flush=True)

    mpra = pd.read_table(args.table_s2, sep='\t', header=0, low_memory=False)
    n_total = mpra.shape[0]

    mpra = mpra.loc[mpra['sequence'].str.len() == VARIABLE_REGION_LEN]
    n_after_len = mpra.shape[0]

    mpra['chr'] = mpra['chr'].astype(str)
    covered = mpra['chr'].isin(fold_index)
    n_uncovered = int((~covered).sum())
    dropped_chrs = sorted(mpra.loc[~covered, 'chr'].unique())
    mpra = mpra.loc[covered].reset_index(drop=True)

    print(f'rows: {n_total} total '
          f'-> {n_after_len} after length == {VARIABLE_REGION_LEN} '
          f'-> {mpra.shape[0]} on covered chromosomes', flush=True)
    print(f'dropped {n_uncovered} rows on chromosomes with no held-out fold: {dropped_chrs}',
          flush=True)
    assert mpra.shape[0] > 0, "no sequences left to score"

    rng = np.random.default_rng(args.seed)
    frames = []
    for chrom, group in mpra.groupby('chr', sort=True):
        if args.sample is not None and group.shape[0] > args.sample:
            group = group.iloc[rng.choice(group.shape[0], args.sample, replace=False)]

        ensemble, n_models = load_fold_models(args.export, config, fold_index[chrom], args.device)
        preds = ensemble.predict(group['sequence'].tolist(), batch_size=args.batch_size,
                                 rc_average=True, device=args.device)
        assert preds.shape == (group.shape[0], len(CELL_TYPES)), \
            f"prediction shape {tuple(preds.shape)} does not match {group.shape[0]} sequences"

        keep_cols = ['chr'] + [f'{c}_log2FC' for c in CELL_TYPES] \
                             + [f'{c}_lfcSE' for c in CELL_TYPES]
        frame = group[keep_cols].copy()
        for i, cell in enumerate(CELL_TYPES):
            frame[f'{cell}_pred'] = preds[:, i].numpy()
        frames.append(frame)
        print(f'chr{chrom}: {group.shape[0]} seqs, {n_models} models', flush=True)

        del ensemble
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    results = pd.concat(frames, ignore_index=True)

    print(f'\nMPAC ensemble vs empirical MPRA activity, {results.shape[0]} sequences scored.')
    print('The preprint reports r = 0.89 / 0.89 / 0.88 at n = 485,180 / 499,820 / 485,034,')
    print('so the comparable row is whichever filter reproduces those per-cell counts.')

    for label, cutoff, mode in FILTERS:
        stats = report(results, label, cutoff, mode)
        if cutoff is None:
            for cell, (r, _) in stats.items():
                assert r > 0.5, f"{cell} correlation collapsed to {r:.4f}; the export is wrong"

    out_path = os.path.join(args.export, 'verification_predictions.tsv.gz')
    results.to_csv(out_path, sep='\t', index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
