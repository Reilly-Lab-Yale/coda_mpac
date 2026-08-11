# Hugging Face export

Tooling that repackages the CODA/MPAC model artifacts into a Hugging Face Hub
repository. The goal is to remove the barriers in the original distribution
format.

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

https://huggingface.co/saarantras1/MPAC: the 110 MPAC checkpoints as safetensors.
That repository is itself a git repo, so the weights are versioned there; this
directory holds the conversion and verification tooling, which is not shipped to the
Hub apart from `modeling_mpac.py`.

## Verification

| Check | Tool | Result |
| --- | --- | --- |
| Conversion fidelity | `convert.py` (inline) | 110/110 bitwise-identical to the `boda` loader |
| Artifact provenance | `compare_zenodo.py` | 110/110 byte-identical to the Zenodo deposit |
| Inference path | `crosscheck_malinois.py` | agrees with an independent implementation to 5 decimals |
| Model performance | `verify.py` | 0.855 / 0.854 / 0.858 on Table S2 (see strand note below) |

## Strand convention

`predict` reverse-complements the 200 bp insert and re-flanks it in the forward
orientation, matching `src/vcf_predict.py`, which produced the published MPAC
predictions. Flipping the assembled 600 bp construct instead -- what the CODA
tutorial does -- scores ~0.035 higher against Table S2 and reproduces the preprint's
Fig. 1B values, but would desynchronise this model from every published MPAC number.
The convention was chosen for consistency, not accuracy.

`crosscheck_malinois.py` deliberately uses the whole-construct flip, since the
independent reference it compares against was computed that way. It validates the
weights, one-hot encoding, flank construction and forward pass -- everything except
the strand choice.

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
