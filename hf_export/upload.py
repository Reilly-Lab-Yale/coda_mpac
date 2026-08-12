"""
Upload a converted export directory to the Hugging Face Hub.

This is the only script here that touches the network in a way that publishes
anything. It refuses to run without an explicit --confirm flag, and defaults to
a private repo so the card and file layout can be reviewed before anyone else
can see them.

Usage:
    huggingface-cli login
    python upload.py --export <dir> --repo_id <org>/<name> --confirm
"""

import argparse
import os
import shutil
import sys

from huggingface_hub import HfApi

REQUIRED = ['config.json', 'provenance.json']
MODULES = ['modeling_mpac.py', 'modeling_malinois.py']

# Regenerable analysis output that verify.py leaves in the export directory. It
# is large, not a model artifact, and would only go stale on the Hub.
IGNORE = ['verification_predictions.tsv.gz']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', required=True, help='Directory written by convert.py')
    parser.add_argument('--repo_id', required=True, help='Target repo, e.g. Reilly-Lab-Yale/MPAC')
    parser.add_argument('--card', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       'README_card.md'),
                        help='Model card to install as README.md')
    parser.add_argument('--public', action='store_true',
                        help='Create the repo public instead of private')
    parser.add_argument('--confirm', action='store_true',
                        help='Required. Without this the script only reports what it would push.')
    args = parser.parse_args()

    for name in REQUIRED:
        path = os.path.join(args.export, name)
        assert os.path.isfile(path), f"export is missing {name}; run convert.py first"

    assert any(os.path.isfile(os.path.join(args.export, m)) for m in MODULES), \
        f"export has none of {MODULES}; run convert.py first"

    n_crossval = len([
        f for _, _, files in os.walk(os.path.join(args.export, 'crossval'))
        for f in files if f.endswith('.safetensors')
    ])
    has_root_model = os.path.isfile(os.path.join(args.export, 'model.safetensors'))

    # Two valid shapes. The MPAC export must NOT carry a root model: every
    # checkpoint is fold-bound, so a default one would be a default way to leak
    # training data. A single-model export (Malinois) is the mirror image.
    if n_crossval:
        assert n_crossval == 110, f"expected 110 crossval checkpoints, found {n_crossval}"
        assert not has_root_model, \
            "fold export also contains a root model.safetensors; that would become the " \
            "chromosome-agnostic default this layout exists to prevent"
    else:
        assert has_root_model, "export has neither crossval checkpoints nor model.safetensors"

    total_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(args.export) for f in files
        if f not in IGNORE
    )

    print(f'export:    {args.export}')
    print(f'repo:      {args.repo_id} ({"public" if args.public else "private"})')
    print(f'crossval:  {n_crossval} checkpoints')
    print(f'size:      {total_bytes / 1e9:.2f} GB')

    if not args.confirm:
        print('\ndry run; pass --confirm to actually upload', file=sys.stderr)
        return

    shutil.copy(args.card, os.path.join(args.export, 'README.md'))

    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=args.export,
        repo_id=args.repo_id,
        repo_type='model',
        ignore_patterns=IGNORE,
        commit_message='Add the 110 MPAC crossval checkpoints as safetensors',
    )
    print(f'\nuploaded to https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
