---
license: mit
library_name: malinois
tags:
  - biology
  - genomics
  - dna
  - mpra
  - cis-regulatory
pipeline_tag: other
---

# Malinois

Malinois predicts cis-regulatory activity of 200 bp human sequences in K562, HepG2
and SK-N-SH. It is a convolutional network trained on MPRA measurements from 776,474
sequences, and the model behind the CODA sequence-design framework.

**The paper is the source of truth for what this model is and how it was evaluated:**
[Machine-guided design of cell-type-targeting cis-regulatory
elements](https://doi.org/10.1038/s41586-024-08070-z) (Gosai et al., Nature 2024).

This repository holds the published checkpoint (`20211113_021200`), converted to
safetensors from `gs://tewhey-public-data/CODA_resources/` with no retraining or
modification.

For genome-wide variant effect prediction, use
[MPAC](https://huggingface.co/saarantras1/MPAC) instead: it is this architecture
retrained across chromosome folds, so a query can be scored by models that never
saw its chromosome. Malinois trained on all chromosomes except its own held-out
sets (validation chr19, chr21, chrX; test chr7, chr13), so scoring human genomic
sequence with it will overstate accuracy on anything it trained on.

## Usage

```python
from modeling_malinois import MalinoisModel

model = MalinoisModel.from_pretrained("saarantras1/malinois").eval()

preds = model.predict(["ACGT" * 50])   # (n, 3): K562, HepG2, SKNSH
```

Use `predict` rather than calling the model directly: it adds the MPRA vector
context the model was trained with (a bare 200mer is not valid input) and averages
over both strands. Skipping either step returns plausible-looking but wrong numbers
instead of an error.

Note on strands: `predict` reverse-complements the 200 bp insert and re-flanks it in
the forward orientation, following `src/vcf_predict.py` in the upstream code base.
The CODA tutorial notebook instead flips the assembled 600 bp construct, which
scores about 0.035 higher against the training library. Both appear in upstream
code; this repository uses the former so that it agrees with the MPAC release.

## Citation

```bibtex
@article{gosai2024coda,
  title   = {Machine-guided design of cell-type-targeting cis-regulatory elements},
  author  = {Gosai, Sager J. and Castro, Rodrigo I. and Fuentes, Natalia and
             Butts, John C. and Mouri, Kousuke and Alasoadura, Michael and
             Kales, Susan and Nguyen, Thanh Thanh L. and Noche, Ramil R. and
             Rao, Arya S. and Joy, Mary T. and Sabeti, Pardis C. and
             Reilly, Steven K. and Tewhey, Ryan},
  journal = {Nature},
  year    = {2024},
  doi     = {10.1038/s41586-024-08070-z}
}
```

## License

MIT, following the declaration in
[sjgosai/boda2](https://github.com/sjgosai/boda2) that the model, model weights and
architecture code are MIT licensed.
