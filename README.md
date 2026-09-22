This is a modified version of https://github.com/sjgosai/boda2 used in [Identifying non-coding variant effects at scale via machine learning models of cis-regulatory reporter assays](https://www.biorxiv.org/content/10.1101/2025.04.16.648420v1).

This manuscript has several other associated repositories.

## Variant Effect Prediction

To generate MPAC predictions for SNVs, first define your variants in a VCF-like TSV (no VCF header required) with the following tab-separated columns:

| Column | Description |
| --- | --- |
| `CHROM` | Chromosome |
| `POS` | Position |
| `ID` | Variant ID (can be blank or `.` if none) |
| `REF` | Reference allele |
| `ALT` | Alternate allele |
| `QUAL` | Quality (can be `.` if null) |
| `FILTER` | Filter status (can be `.` if null) |
| `INFO` | Can be `.` if null; predictions populate this column in the output |

### SNVs

SNV predictions are handled with `vcf_predict.py`:

```bash
python vcf_predict.py \
    --artifact_path {10X $MODEL} \
    --vcf_file ${VCF} \
    --fasta_file ${FASTA} \
    --output ${OUTPUT} \
    --relative_start 9 \
    --relative_end 180 \
    --step_size 10 \
    --strand_reduction mean \
    --window_reduction mean \
    --feature_ids K562 HepG2 SKNSH
```

| Argument | Description |
| --- | --- |
| `--artifact_path` | Chromosome holdout models to ensemble |
| `--vcf_file` | Variants of interest |
| `--fasta_file` | Reference genome |
| `--output` | Output path |
| `--relative_start` | First window variant position |
| `--relative_end` | Final window variant position |
| `--step_size` | Window step size |
| `--strand_reduction` | Reduction method for fwd/rev strand predictions |
| `--window_reduction` | Reduction method for window predictions |
| `--feature_ids` | Labels for predictions in the output |

Additonal arguments can be found in vcf_predict.py

### Small indels

Small indels (≤ 10 bp recommended) are handled the same way with `vcf_predict_indel.py`, with modification to the plasmid sequence padding loop.

### Haplotypes

Haplotype predictions are handled similarly with `vcf_predict_haplotype.py`, with modification to the windowing loop to ensure all windows contain the desired variants.

### Output

Output is a VCF-like TSV with predictions occupying the `INFO` column, see below:

| chrom | pos | id | ref | alt | INFO |
| chr22 | 11121724 | COSV106573183 | A | G | K562__ref=0.3535322;HepG2__ref=0.29076257;SKNSH__ref=0.46344832;K562__alt=0.3156159;HepG2__alt=0.27474153;SKNSH__alt=0.4554421;K562__skew=-0.03791629;HepG2__skew=-0.016021034;SKNSH__skew=-0.0080062505 |
