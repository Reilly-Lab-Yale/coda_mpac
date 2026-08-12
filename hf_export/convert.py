"""
Convert MPAC model artifacts into a Hugging Face Hub layout.

Reads the `.tar.gz` artifacts (or already-unpacked `torch_checkpoint.pt` files)
produced by `src/train.py`, normalizes their hyperparameters into a single
`config.json`, and writes weights as safetensors. Every converted checkpoint is
verified against the original `boda` loader before it is accepted.

Output layout:

    <out>/config.json                       normalized architecture config
    <out>/modeling_mpac.py                  standalone model definition
    <out>/crossval/test_<a>_<b>/val_<c>_<d>.safetensors
    <out>/provenance.json                   per-file source path and chr splits

There is deliberately no default `model.safetensors` at the root: every MPAC
checkpoint is tied to a chromosome fold, so naming one of them "the" model would
invite exactly the training-data leak the cross-validation exists to prevent.

Usage:
    python convert.py --crossval_root <dir> --out <dir>
"""

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpac_hf.modeling_mpac import CELL_TYPES, MPACModel

# Hyperparameters that describe the architecture. Everything else in a
# checkpoint's `model_hparams` is training-time state (loss scaling, reduction
# mode) that the inference model has no use for.
ARCH_KEYS = [
    'input_len',
    'conv1_channels', 'conv1_kernel_size',
    'conv2_channels', 'conv2_kernel_size',
    'conv3_channels', 'conv3_kernel_size',
    'n_linear_layers', 'linear_channels', 'linear_activation', 'linear_dropout_p',
    'n_branched_layers', 'branched_channels', 'branched_activation', 'branched_dropout_p',
    'n_outputs', 'use_batch_norm', 'use_weight_norm',
]


def infer_input_len(state_dict, conv3_channels):
    """Recover input_len from the first linear layer's fan-in.

    The 2023 crossval checkpoints omit `input_len` from their saved hparams and
    rely on the constructor default. Rather than trust that default, invert
    `get_flatten_factor`: flatten_factor = (input_len // 12 + 2) // 4.
    """
    in_features = state_dict['linear1.linear.weight'].shape[1]
    assert in_features % conv3_channels == 0, \
        f"linear1 fan-in {in_features} not divisible by conv3_channels {conv3_channels}"
    flatten_factor = in_features // conv3_channels
    input_len = 12 * (4 * flatten_factor - 2)
    return input_len


def normalize_hparams(checkpoint):
    """Extract a version-independent architecture config from a checkpoint."""
    hparams = checkpoint['model_hparams']
    hparams = hparams if isinstance(hparams, dict) else vars(hparams)
    state_dict = checkpoint['model_state_dict']

    assert checkpoint['model_module'] == 'BassetBranched', \
        f"only BassetBranched is supported, got {checkpoint['model_module']}"

    config = {k: hparams[k] for k in ARCH_KEYS if k in hparams}

    inferred = infer_input_len(state_dict, config['conv3_channels'])
    if 'input_len' in config:
        assert config['input_len'] == inferred, \
            f"declared input_len {config['input_len']} disagrees with weights ({inferred})"
    else:
        config['input_len'] = inferred

    config['variable_region_len'] = 200
    if config['n_outputs'] == len(CELL_TYPES):
        config['output_names'] = CELL_TYPES

    return config


def load_checkpoint(path, workdir):
    """Load a torch_checkpoint dict from a .tar.gz artifact or a direct .pt path."""
    if tarfile.is_tarfile(path):
        shutil.unpack_archive(path, workdir)
        pt_path = os.path.join(workdir, 'artifacts', 'torch_checkpoint.pt')
        assert os.path.isfile(pt_path), f"no artifacts/torch_checkpoint.pt inside {path}"
    else:
        pt_path = path
    return torch.load(pt_path, map_location='cpu', weights_only=False)


def check_parity(checkpoint, config, n_seq=4, seed=0):
    """Assert the standalone model reproduces the original boda model bitwise.

    Builds the model both ways and compares logits on the same random one-hot
    input. A silent architecture transcription error is the main risk in this
    conversion, and it would not show up as a load failure.
    """
    import boda

    model_module = getattr(boda.model, checkpoint['model_module'])
    hparams = checkpoint['model_hparams']
    reference = model_module(**(hparams if isinstance(hparams, dict) else vars(hparams)))
    reference.load_state_dict(checkpoint['model_state_dict'])
    reference.eval()

    ported = MPACModel(**config)
    ported.load_state_dict(checkpoint['model_state_dict'])
    ported.eval()

    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n_seq, config['input_len']), generator=generator)
    x = torch.nn.functional.one_hot(idx, num_classes=4).permute(0, 2, 1).float()

    with torch.no_grad():
        ref_out = reference(x)
        new_out = ported(x)

    assert torch.equal(ref_out, new_out), (
        f"port diverges from boda reference: max abs diff "
        f"{(ref_out - new_out).abs().max().item():.3e}"
    )
    return ported


def parse_bucket_map(pull_script):
    """Map artifact basename -> original gs:// URL, from pull_models_home.py."""
    if pull_script is None or not os.path.isfile(pull_script):
        return {}
    with open(pull_script) as handle:
        urls = re.findall(r"gs://[^'\"\s]+\.tar\.gz", handle.read())
    mapping = {os.path.basename(u): u for u in urls}
    assert len(mapping) == len(urls), "duplicate artifact basenames in bucket listing"
    return mapping


def chr_split(checkpoint):
    """Pull the train/val/test chromosome split out of the saved data hparams."""
    data_hparams = checkpoint.get('data_hparams')
    if data_hparams is None:
        return {}
    data_hparams = data_hparams if isinstance(data_hparams, dict) else vars(data_hparams)
    return {
        'val_chrs': data_hparams.get('val_chrs'),
        'test_chrs': data_hparams.get('test_chrs'),
    }


def read_artifact(src):
    """Load a checkpoint into memory, unpacking to a scratch dir if needed."""
    with tempfile.TemporaryDirectory() as workdir:
        return load_checkpoint(src, workdir)


def convert_one(checkpoint, src, dest, bucket_map, config_ref):
    """Convert a loaded checkpoint to safetensors, returning its provenance record."""
    config = normalize_hparams(checkpoint)

    if config_ref is not None:
        assert config == config_ref, (
            f"{os.path.basename(src)} has a different architecture than the "
            f"reference; a single shared config.json would be wrong.\n"
            f"  reference: {config_ref}\n  this file: {config}"
        )

    check_parity(checkpoint, config)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    state_dict = {k: v.contiguous() for k, v in checkpoint['model_state_dict'].items()}
    save_file(state_dict, dest, metadata={'format': 'pt'})

    record = {
        'file': dest,
        'source': bucket_map.get(os.path.basename(src), src),
        'timestamp': checkpoint.get('timestamp'),
        'random_tag': checkpoint.get('random_tag'),
    }
    record.update(chr_split(checkpoint))
    return config, record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--crossval_root',
                        help='Directory of chr<a>_<b>/ subdirs holding crossval artifacts')
    parser.add_argument('--single',
                        help='Convert one artifact to model.safetensors at the export root, '
                             'for a single-model repo such as the original Malinois. '
                             'Mutually exclusive with --crossval_root.')
    parser.add_argument('--module_name', default='modeling_mpac.py',
                        help='Filename to give the model definition in the export')
    parser.add_argument('--source_url',
                        help='Canonical origin to record in provenance.json for --single, '
                             'when the artifact is not covered by --pull_script')
    parser.add_argument('--pull_script', default=None,
                        help='pull_models_home.py, used to recover gs:// source URLs')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--limit', type=int, default=None,
                        help='Convert at most this many crossval models (for smoke tests)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bucket_map = parse_bucket_map(args.pull_script)

    assert bool(args.crossval_root) != bool(args.single), \
        "pass exactly one of --crossval_root or --single"

    config = None
    records = []

    if args.single:
        print(f'[single] {args.single}', flush=True)
        config, record = convert_one(
            read_artifact(args.single), args.single,
            os.path.join(args.out, 'model.safetensors'), bucket_map, None
        )
        record['role'] = 'default'
        if args.source_url:
            record['source'] = args.source_url
        records.append(record)

    folds = sorted(d for d in os.listdir(args.crossval_root)
                   if os.path.isdir(os.path.join(args.crossval_root, d))) \
        if args.crossval_root else []
    n_done = 0
    for fold in folds:
        fold_dir = os.path.join(args.crossval_root, fold)
        for name in sorted(os.listdir(fold_dir)):
            if not name.endswith('.tar.gz'):
                continue
            if args.limit is not None and n_done >= args.limit:
                break
            src = os.path.join(fold_dir, name)
            checkpoint = read_artifact(src)
            split = chr_split(checkpoint)
            assert '_'.join(split['test_chrs']) == fold.replace('chr', ''), (
                f"{src} is in fold {fold} but its test chromosomes are "
                f"{split['test_chrs']}"
            )
            val_tag = '_'.join(split['val_chrs'])
            dest = os.path.join(args.out, 'crossval', f'test_{fold.replace("chr", "")}',
                                f'val_{val_tag}.safetensors')
            assert not os.path.exists(dest), f"two checkpoints map to {dest}"
            # The first checkpoint defines the shared architecture; the rest are
            # asserted against it inside convert_one.
            this_config, record = convert_one(checkpoint, src, dest, bucket_map, config)
            config = config if config is not None else this_config
            record['role'] = 'crossval'
            record['fold'] = fold
            records.append(record)
            n_done += 1
            print(f'[crossval {n_done}] {fold}/{name} -> {os.path.relpath(dest, args.out)}',
                  flush=True)

    assert records, "no artifacts converted"

    with open(os.path.join(args.out, 'config.json'), 'w') as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    for record in records:
        record['file'] = os.path.relpath(record['file'], args.out)
    with open(os.path.join(args.out, 'provenance.json'), 'w') as handle:
        json.dump(records, handle, indent=2)

    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'mpac_hf', 'modeling_mpac.py'),
                os.path.join(args.out, args.module_name))

    n_missing = sum(1 for r in records if not r['source'].startswith('gs://'))
    print(f'\nconverted {len(records)} checkpoints into {args.out}')
    print(f'config: {config}')
    if n_missing:
        print(f'warning: {n_missing} record(s) have no gs:// source URL', file=sys.stderr)


if __name__ == '__main__':
    main()
