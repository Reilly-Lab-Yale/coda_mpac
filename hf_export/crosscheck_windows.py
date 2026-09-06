"""Check the Hub windowing path against `src/vcf_predict.py`.

`MPACModel.predict_windows` / `predict_skew` reimplement the sliding-window scheme
that produced the published MPAC predictions. This scores the same variants both
ways -- once through the real pipeline, once through the Hub module -- and reports
the disagreement.

The two paths are not bitwise comparable: `vcf_predict.py` runs the model under
`torch.cuda.amp.autocast()` (VepTester, ~line 201) while `predict` is fp32, worth
about 1e-2. The run therefore compares twice, fp32 and under matching autocast; the
second should collapse to near zero if the window geometry and reductions agree.

Usage:
    PYTHONPATH=<repo root> python crosscheck_windows.py --export <dir> \\
        --fasta <GRCh38.fasta> --models <chr7_16 dir> [--n_variants 200]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpac_hf.modeling_mpac import (CELL_TYPES, MPAC_CONTEXT_DOWNSTREAM,
                                   MPAC_CONTEXT_UPSTREAM, MPAC_STEP_SIZE,
                                   MPACEnsemble, MPACModel)

RELATIVE_START, RELATIVE_END = 9, 181
CONTEXT = MPAC_CONTEXT_UPSTREAM + 1 + MPAC_CONTEXT_DOWNSTREAM
N_WINDOWS = 18


def sample_variants(fasta_path, chrom, n, seed):
    """Pick SNV positions whose whole context is unambiguous uppercase ACGT.

    Soft-masked (lowercase) and N-containing regions are skipped so the comparison
    tests window geometry rather than each path's case/ambiguity handling.
    """
    fasta = Fasta(fasta_path, sequence_always_upper=False)
    contig = fasta[chrom]
    rng = np.random.default_rng(seed)

    rows, tries = [], 0
    while len(rows) < n and tries < n * 200:
        tries += 1
        pos = int(rng.integers(CONTEXT, len(contig) - CONTEXT))       # 1-based
        var_loc = pos - 1
        start = var_loc - MPAC_CONTEXT_UPSTREAM
        ctx = str(contig[start:start + CONTEXT])
        if len(ctx) != CONTEXT or set(ctx) - set('ACGT'):
            continue
        ref = ctx[MPAC_CONTEXT_UPSTREAM]
        alt = rng.choice([b for b in 'ACGT' if b != ref])
        rows.append({'chrom': chrom, 'pos': pos, 'id': f'{chrom}:{pos}:{ref}:{alt}',
                     'ref': ref, 'alt': alt, 'ref_ctx': ctx,
                     'alt_ctx': ctx[:MPAC_CONTEXT_UPSTREAM] + alt
                                + ctx[MPAC_CONTEXT_UPSTREAM + 1:]})
    assert len(rows) == n, f"only found {len(rows)} clean sites after {tries} tries"

    variants = pd.DataFrame(rows)
    assert variants['pos'].is_unique, "duplicate variant positions sampled"
    assert (variants['ref_ctx'].str.len() == CONTEXT).all()
    assert (variants['alt_ctx'].str.len() == CONTEXT).all()
    assert (variants['ref_ctx'] != variants['alt_ctx']).all()
    return variants


def write_vcf(variants, path):
    with open(path, 'w') as handle:
        print('#CHROM\tPOS\tID\tREF\tALT', file=handle)
        for _, v in variants.iterrows():
            print(f"{v['chrom']}\t{v['pos']}\t{v['id']}\t{v['ref']}\t{v['alt']}", file=handle)


def run_vcf_predict(repo_root, models_dir, vcf_path, fasta_path, out_path):
    artifacts = sorted(os.path.join(models_dir, f) for f in os.listdir(models_dir)
                       if f.endswith('.tar.gz'))
    assert len(artifacts) == 10, f"expected 10 fold artifacts, found {len(artifacts)}"
    cmd = [sys.executable, os.path.join(repo_root, 'src', 'vcf_predict.py'),
           '--artifact_path', *artifacts,
           '--use_vmap', 'TRUE',
           '--vcf_file', vcf_path, '--fasta_file', fasta_path, '--output', out_path,
           '--relative_start', str(RELATIVE_START), '--relative_end', str(RELATIVE_END),
           '--step_size', str(MPAC_STEP_SIZE),
           '--strand_reduction', 'mean', '--window_reduction', 'mean',
           '--raw_predictions', 'FALSE',
           '--feature_ids', *CELL_TYPES]
    print('+ ' + ' '.join(cmd[:6]) + ' ... (10 artifacts)', flush=True)
    subprocess.run(cmd, check=True, cwd=repo_root,
                   env={**os.environ, 'PYTHONPATH': repo_root})

    table = pd.read_table(out_path)
    parsed = {}
    for field in table['INFO']:
        for kv in field.split(';'):
            key, value = kv.split('=')
            parsed.setdefault(key, []).append(float(value))
    out = pd.DataFrame(parsed)
    assert out.shape[0] == table.shape[0], "INFO parse dropped rows"
    return out


def hub_predictions(export, models_dir, variants, device, autocast):
    import json
    config = json.load(open(os.path.join(export, 'config.json')))
    fold = os.path.basename(models_dir).replace('chr', 'test_')
    fold_dir = os.path.join(export, 'crossval', fold)

    models = []
    for name in sorted(os.listdir(fold_dir)):
        model = MPACModel(**config)
        model.load_state_dict(load_file(os.path.join(fold_dir, name)))
        models.append(model.eval().to(device))
    ensemble = MPACEnsemble(models).to(device)

    kwargs = dict(batch_size=256, device=device)
    if autocast:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = ensemble.predict_skew(variants['ref_ctx'].tolist(),
                                        variants['alt_ctx'].tolist(), **kwargs)
    else:
        out = ensemble.predict_skew(variants['ref_ctx'].tolist(),
                                    variants['alt_ctx'].tolist(), **kwargs)
    return {k: v.float() for k, v in out.items()}


def compare(label, boda, hub, n_variants):
    print(f'\n{label}')
    print(f'{"field":14s} {"max|d|":>10s} {"mean|d|":>10s} {"pearson":>10s}')
    worst = 0.0
    for tag in ['ref', 'alt', 'skew']:
        for i, cell in enumerate(CELL_TYPES):
            b = boda[f'{cell}__{tag}'].to_numpy()
            h = hub[tag][:, i].numpy()
            assert b.shape == h.shape == (n_variants,), \
                f"{cell}__{tag}: {b.shape} vs {h.shape}"
            d = np.abs(b - h)
            r = np.corrcoef(b, h)[0, 1]
            worst = max(worst, d.max())
            print(f'{cell}__{tag:5s} {d.max():10.2e} {d.mean():10.2e} {r:10.6f}')
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', required=True)
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--models', required=True, help='chr<a>_<b> dir of 10 .tar.gz')
    parser.add_argument('--repo_root', default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--chrom', default='chr7')
    parser.add_argument('--n_variants', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    variants = sample_variants(args.fasta, args.chrom, args.n_variants, args.seed)
    print(f'{len(variants)} SNVs on {args.chrom}, {CONTEXT} bp context, '
          f'{N_WINDOWS} windows of 200 bp at stride {MPAC_STEP_SIZE}', flush=True)

    vcf_path = os.path.join(args.workdir, 'crosscheck.vcf')
    write_vcf(variants, vcf_path)

    boda = run_vcf_predict(args.repo_root, args.models, vcf_path, args.fasta,
                           os.path.join(args.workdir, 'crosscheck_boda.vcf'))

    worst_fp32 = compare('hub fp32 vs vcf_predict.py (autocast fp16)',
                         boda, hub_predictions(args.export, args.models, variants,
                                               device, autocast=False),
                         len(variants))
    worst_amp = compare('hub autocast fp16 vs vcf_predict.py (autocast fp16)',
                        boda, hub_predictions(args.export, args.models, variants,
                                              device, autocast=True),
                        len(variants))

    print(f'\nworst |d|: fp32 {worst_fp32:.2e}, matched autocast {worst_amp:.2e}')
    assert worst_amp < 1e-2, \
        f"windowing paths disagree beyond precision noise: {worst_amp:.3e}"
    print('windowing and reductions agree with vcf_predict.py')


if __name__ == '__main__':
    main()
