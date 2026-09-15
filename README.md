# Fair-Medical-Imaging

## Download CheXpert

Dataset access is handled by Stanford. Please register for an account and download the CheXpert demographics data [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf). 

## Prepare the data

The preparation script creates **balanced training and validation sets** and a
**test set that retains its natural imbalance**, with no patient overlap between
splits.

The four binary tasks are **No Finding**, **Cardiomegaly**, **Pleural Effusion**
(`Effusion` in the code), and **Pneumothorax**. Demographic labels are grouped as:

| Attribute | Groups |
| --- | --- |
| Age | < 60 years / ≥ 60 years |
| Sex | Female / Male |
| Race/ethnicity | White / Non-White |

These attributes define eight intersectional groups. The preparation script
includes unknown race in the Non-White/other group.

Run from the repository root:

```bash
python prepare_chexpert_data.py \
  --chexpert-root data/CheXpert-v1.0 \
  --demographics data/chexpert_demographics.xlsx \
  --output data/processed
```

The combined task CSVs are `no_finding.csv`, `cardiomegaly.csv`, `effusion.csv`,
and `pneumothorax.csv` under `data/processed/`. Each contains train, val, and test
rows and can be passed directly to `--metadata_csv`. Separate balanced train/val
and natural test tables are also saved in `data/processed/train/` and
`data/processed/test/`.

## Train

Each run trains one binary disease classifier:

```bash
python -m training.train \
  --target Pneumothorax \
  --metadata_csv data/processed/pneumothorax.csv \
  --image_root data/CheXpert-v1.0 \
  --output_dir outputs/pneumothorax \
  --epochs 40 \
  --batch_size 128 \
  --balanced_batches
```

## Evaluate

### Disease prediction

Training automatically evaluates four validation-selected checkpoints:

| Checkpoint | Selection criterion |
| --- | --- |
| `best_val_loss` | Classification loss |
| `best_val_auc` | AUC |
| `best_val_delta_eo` | Intersectional equalized-odds disparity |
| `best_val_delta_auc` | Intersectional AUC disparity |


To evaluate existing checkpoints, rerun your training command with
`--evaluation_only`.

### Frozen representation probes

Fit probes to predict disease and demographic labels from each learned space:

```bash
python -m evaluation.evaluate_four_directions \
  --task_dir outputs/pneumothorax \
  --checkpoint best_val_auc \
  --metadata_csv data/processed/pneumothorax.csv \
  --image_root data/CheXpert-v1.0
```

### FATE and DRAR

`evaluation/metrics.py` contains the shared metric functions and the command-line
entry point for BACC, Macro F1, AUC, intersectional disparities, FATE, and DRAR.
**DRAR is CKA Reduction relative to ERM.**

Prepare three `.npz` exports:

| Export | Required arrays |
| --- | --- |
| Method | `sample_ids` `[n]`, `labels` `[n]`, `attributes` `[n,3]`, `scores` `[n]`, `z_d` `[n,h]` |
| ERM | The same fields, with ERM predictions and representations |
| Attribute reference | `sample_ids` `[n]`, `z_a_ref` `[n,h_a]` from a separately supervised attribute encoder |

```bash
python -m evaluation.metrics \
  --method exports/method.npz \
  --erm exports/erm.npz \
  --attribute-reference exports/attribute_reference.npz \
  --task Pneumothorax \
  --output outputs/pneumothorax/paper_metrics
```

## Repository layout

```
prepare_chexpert_data.py        # Raw tables → patient-disjoint task CSVs

training/
├── train.py                    # Training and checkpoint evaluation
├── model.py                    # Shared encoder and representation heads
├── losses.py                   # Training objectives
├── data.py                     # Metadata, transforms, and sampling
├── disease_pcgrad.py           # Gradient coordination
└── multistate_adamw.py          # Disease-priority optimizer

evaluation/
├── metrics.py                  # Metric functions include FATE/DRAR evaluation
└── evaluate_representations.py  # Frozen representation probes
```
