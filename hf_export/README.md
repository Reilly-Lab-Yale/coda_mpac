# Hugging Face export

Tooling that repackages the CODA/MPAC model artifacts into a Hugging Face Hub
repository. The goal is to remove the barriers in the original distribution
format: a `.tar.gz` on a GCS bucket, holding a checkpoint whose hyperparameters
are a pickled `argparse.Namespace`, loadable only with `weights_only=False` and
an install of `boda` pinned to `lightning==1.9.5`.

The export depends on `torch`, `safetensors`, and `huggingface_hub` only.

## Contents

| File | Purpose |
| --- | --- |
| `mpac_hf/modeling_mpac.py` | Standalone `BassetBranched` definition. No `boda`, no `lightning`. Ships to the Hub alongside the weights. |
| `convert.py` | Artifacts to safetensors, with a normalized `config.json` and a bitwise parity check against the original loader. |
| `verify.py` | End-to-end check: reproduces predicted-vs-empirical MPRA correlation using the correct held-out fold per chromosome. |
| `upload.py` | Pushes an export directory to the Hub. Requires `--confirm`; private by default. |
| `README_card.md` | Model card, installed as `README.md` at upload time. |
| `compare_zenodo.py` | CRC-32 check of the local artifacts against the published Zenodo zip. |
| `crosscheck_malinois.py` | Compares the inference path against an independent implementation on chr7+13. |
| `verify_gpu.sh` | Slurm wrapper used to run `verify.py` on a GPU node (Bouchet-specific paths). |

`compare_zenodo.py`, `crosscheck_malinois.py` and `verify_gpu.sh` carry absolute
paths for this cluster as module-level constants; edit them before reuse elsewhere.

## Published model

https://huggingface.co/saarantras1/MPAC -- the 110 MPAC checkpoints as safetensors.
That repository is itself a git repo, so the weights are versioned there; this
directory holds the conversion and verification tooling, which is not shipped to the
Hub apart from `modeling_mpac.py`.

## What was verified

| Check | Tool | Result |
| --- | --- | --- |
| Conversion fidelity | `convert.py` (inline) | 110/110 bitwise-identical to the `boda` loader |
| Artifact provenance | `compare_zenodo.py` | 110/110 byte-identical to the Zenodo deposit |
| Inference path | `crosscheck_malinois.py` | agrees with an independent implementation to 5 decimals |
| Model performance | `verify.py` | 0.892 / 0.888 / 0.879 vs 0.89 / 0.89 / 0.88 in the preprint |

The inference-path cross-check is the one that matters most: bitwise weight parity
does not cover flanking or reverse-complement handling, and a bug there cost about
0.05 Pearson while still looking plausible. Re-run it after any change to `predict`.

## Running it

```bash
python convert.py \
    --crossval_root <dir of chr<a>_<b>/ subdirs> \
    --pull_script   <pull_models_home.py> \
    --out           <export dir>

python verify.py --export <export dir> --table_s2 <Table_S2__MPRA_dataset.txt>

python upload.py --export <export dir> --repo_id <org>/<name> --confirm
```

`convert.py` refuses to proceed if any checkpoint's architecture differs from
the first one converted, since the export publishes a single shared
`config.json`. It also recovers `input_len` from the weights rather than from
the saved hyperparameters, because the 2023 crossval checkpoints omit that key
and rely on a constructor default.

## Notes on the source artifacts

The 110 MPAC checkpoints hold out pairs of autosomes whose numbers sum to 23
(chr1+chr22 through chr11+chr12), with ten replicates per fold differing in
which pair was used for validation. Predicting on a human sequence requires the
fold whose *test* chromosomes contain that sequence's source chromosome; using
any other fold leaks training data. chrX and chrY are not covered by any fold.

The export publishes no root `model.safetensors`: with every checkpoint tied to a
fold, a default model would amount to a default way to leak training data.
`MPACEnsemble.from_pretrained(repo_id, chromosome=N)` resolves the right fold and
refuses chromosomes that no fold held out.
