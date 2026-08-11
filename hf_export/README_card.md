---
license: cc-by-4.0
library_name: mpac
tags:
  - biology
  - genomics
  - dna
  - mpra
  - cis-regulatory
  - variant-effect-prediction
pipeline_tag: other
---

# MPAC

MPAC (Malinois with Parallel Aggregated Cross-validation) predicts cis-regulatory
activity of 200 bp human sequences in K562, HepG2 and SK-N-SH, and the allelic skew
caused by non-coding variants.

[Identifying non-coding variant effects at scale via machine learning models of
cis-regulatory reporter assays](https://doi.org/10.1101/2025.04.16.648420).

This repository holds the 110 published checkpoints, converted to safetensors from
the [Zenodo deposit](https://doi.org/10.5281/zenodo.15178434) with no retraining or
modification.

## Usage

```python
from modeling_mpac import MPACEnsemble

# Loads the ten models that held chr7 out of training
ensemble = MPACEnsemble.from_pretrained("saarantras1/MPAC", chromosome=7, device="cuda")

preds = ensemble.predict(["ACGT" * 50], device="cuda")   # (n, 3): K562, HepG2, SKNSH
```

Use `from_pretrained` and `predict` rather than loading a checkpoint or calling the
model directly: they select the ensemble that did not train on your query's
chromosome, and they add the MPRA vector context and average over both strands.
Skipping either step returns plausible-looking but wrong numbers instead of an error.

`predict` follows `vcf_predict.py` from the upstream code base, which generated the
published predictions: the reverse strand is the reverse complement of the 200 bp
insert placed back in the forward-orientation vector, matching the assay, rather
than a reverse complement of the whole 600 bp construct.

MPAC covers autosomes only; `from_pretrained` raises on chrX, chrY and anything else
with no held-out fold.

For variant-effect prediction, see
[john-c-butts/MPAC](https://github.com/john-c-butts/MPAC).

## Citation

```bibtex
@article{butts2025mpac,
  title   = {Identifying non-coding variant effects at scale via machine learning
             models of cis-regulatory reporter assays},
  author  = {Butts, John C. and Rong, Stephen and Gosai, Sager J. and
             Castro, Rodrigo I. and Noon, Mackenzie and Adeniran, Kehinde and
             Ghosh, Rohit and Sabeti, Pardis C. and Tewhey, Ryan and Reilly, Steven K.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.04.16.648420}
}
```

## License

CC-BY-4.0, matching the Zenodo deposit. `modeling_mpac.py` derives from the MIT
licensed model code in [sjgosai/boda2](https://github.com/sjgosai/boda2) and retains
that notice.
