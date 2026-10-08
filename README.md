# DML-project

Bebbe får hela jävla johannebergs berggrund att krackelera när han pullar upp till campus #true

Länkar:
Projek Instruc: https://canvas.chalmers.se/courses/40843/pages/project-instructions?module_item_id=706491

Data Sets:
PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/

Workflow after dataloader and datasets are constructed

Step 1 CNN model class Conv1d with 12 inputs, 5 outputs; test shapes with a fake batch
Step 2 Training loop multi-label loss + macro AUC, with an optional lead-dropout switch
Step 3 Train model A dropout off; save the trained weights to disk
Step 4 Evaluation with masks test A on validation: all 12 leads vs. 3 leads
Step 5 Train model B the same code with dropout on; save the weights
Step 6 Compare on validation choose the dropout rate here, never on test
Step 7 Final test runs A and B on 12 leads, on I/II/V2, and on all 220 triplets
Step 8 Analysis and plots which leads matter, which diagnoses suffer most

Theodors catchup


# ECG Classification with Reduced Leads — Project README

Status summary so the whole group is on the same page. Last updated: 2026-10-08.
The preprocessing code lives in the repo; this file explains what it does and why.

---

## 1. Project goal

Train a deep neural network (PyTorch) to classify 12-lead ECG recordings, then test how well it works when **only 3 leads are available** (the other 9 set to zero).

The core experiment compares two models with **identical architecture and settings**:

| Model | Trained with | Tested on |
|---|---|---|
| **A** | all 12 leads | 12 leads, and 3 leads (rest zeroed) |
| **B** | **lead dropout** (random leads zeroed during training) | 12 leads, and 3 leads (rest zeroed) |

Hypothesis: A degrades badly when leads are missing; B degrades gracefully because it never learned to rely on any single lead.

This is a proof of concept. We reuse the same test set for all lead configurations, but **all choices (hyperparameters, best epoch, dropout rate, "best" lead set) are made on the validation set, never on test.**

---

## 2. Dataset: PTB-XL (v1.0.1)

- Source: PhysioNet, *PTB-XL, a large publicly available electrocardiography dataset* (Wagner et al., 2020). Licence CC BY 4.0. Cite both the PhysioNet entry and the Scientific Data paper.
- 21,837 clinical 12-lead ECGs, 10 s each, from 18,885 patients.
- Two sampling rates are provided; **we use 100 Hz** (`records100/`), so each record is 1000 samples × 12 leads.
- Signals are stored in WFDB format (`.dat` + `.hea` per record) in millivolts, read with the `wfdb` Python package.

### Folder layout (ours)

```
datav2/ptbxl/
├── ptbxl_database.csv      # one row per record: patient info, labels, fold, file paths
├── scp_statements.csv      # maps SCP codes -> diagnostic superclass
├── records100/             # 100 Hz waveforms  (used)
├── records500/             # 500 Hz waveforms  (not used yet)
├── X100.npy                # OUR CACHE of all 100 Hz signals, (21837, 1000, 12), ~1 GB
└── norm_stats.npz          # OUR per-lead mean/std from the training set
```

### Key columns in `ptbxl_database.csv` (loaded as `meta`)

| Column | Meaning | Used for |
|---|---|---|
| `ecg_id` (index) | record ID | row identity |
| `patient_id` | patient ID | leakage check |
| `scp_codes` | dict `{code: likelihood}`, e.g. `{'NORM': 100.0, 'SR': 0.0}` | labels |
| `strat_fold` | 1–10 | train/val/test split |
| `filename_lr` | path to the 100 Hz record | loading signals |
| `age`, `sex`, `height`, `weight`, ... | patient/recording metadata | analysis only (not model input yet) |

Notes: `sex` is 0 = male, 1 = female. Ages > 89 are stored as 300 (privacy). Height/weight are often missing. A likelihood of `0.0` in `scp_codes` means "unknown", not "absent".

### Labels: 5 diagnostic superclasses (multi-label)

Each record can have **several** superclasses at once. SCP codes are mapped to superclasses with `scp_statements.csv` (only rows with `diagnostic == 1`; rhythm/form codes like `SR`, `SBRAD`, `LVOLT` are ignored).

| Superclass | Meaning | Records |
|---|---|---|
| NORM | Normal ECG | 9,528 |
| MI | Myocardial infarction | 5,486 |
| STTC | ST/T change | 5,250 |
| CD | Conduction disturbance | 4,907 |
| HYP | Hypertrophy | 2,655 |

Superclasses per record: 1 → 16,272 · 2 → 4,079 · 3 → 920 · 4 → 159 · **0 → 407 (dropped)**.
We chose **multi-label** classification (the standard PTB-XL task) rather than keeping only single-label records, which would discard ~25% of the data.

### The 12 leads

Channel order (as in the WFDB headers): `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.

Important for the reduced-lead part: the 6 limb leads come from the same 3 electrodes, so **III, aVR, aVL, aVF can be computed exactly from I and II** (e.g. III = II − I). There are effectively only 8 independent leads (I, II, V1–V6). A 3-lead set like **I, II, V2** carries more information than I, II, III.

---

## 3. Preprocessing pipeline (done)

| Step | What | Result |
|---|---|---|
| 1 | Load `ptbxl_database.csv`, parse `scp_codes` text into dicts | `meta`: (21837, 27) |
| 2 | Map codes → list of superclasses | new column `meta["superclass"]` |
| 3 | Load all waveforms, cache to `X100.npy` | `X`: (21837, 1000, 12) float32 |
| 4 | Drop 407 records with no superclass (same mask on `X` and `meta`) | (21430, ...) |
| 5 | Split by `strat_fold`: 1–8 train, 9 val, 10 test | 17,111 / 2,156 / 2,163 |
| 6 | Labels → multi-hot vectors (`MultiLabelBinarizer`) | `y_*`: (N, 5) float32 |
| 7 | Standardize each lead with **training-set** mean/std | train ≈ mean 0, std 1 per lead |
| 8 | Transpose to channels-first for `Conv1d` | (N, 12, 1000) |
| 9 | `ECGDataset` + `DataLoader` (batch 64 train, 256 val/test) | batches of (64, 12, 1000) and (64, 5) |

### Why the splits are trustworthy

The `strat_fold` folds were made by the dataset authors: stratified by diagnosis/age/sex, and **all records of one patient are in the same fold** (no patient leakage). Folds 9 and 10 were checked by at least one human cardiologist, which is why they are the recommended validation/test folds. Using them also makes our numbers comparable with published PTB-XL results.

### Normalization facts

Raw per-lead means are ~0 mV (baseline-centered). Raw stds are ~0.14–0.16 mV for limb leads and ~0.23–0.33 mV for chest leads V1–V6 (closer to the heart, bigger signal). That is why we normalize **per lead**, using training statistics only (applied unchanged to val/test). After normalization, 0 = the lead's average, so a zeroed lead acts as "missing / no information", which is what lead dropout and lead masking rely on.

### Class label order

`MultiLabelBinarizer` sorts alphabetically: **`['CD', 'HYP', 'MI', 'NORM', 'STTC']`**. Example: `['NORM']` → `[0, 0, 0, 1, 0]`, `['CD', 'MI']` → `[1, 0, 1, 0, 0]`. Use `CLASS_NAMES` for plot/report labels.

### Batch sizes

64 for training is a hyperparameter (affects learning). Validation/test use 256 only for speed; evaluation results do not depend on batch size as long as `model.eval()` is called.

---

## 4. Setup and practical notes

- Conda environment `dml` with `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch` and `wfdb`. After installing a package from a notebook, restart the kernel.
- The first load reads ~43,000 small files and takes several minutes. After that, `X100.npy` loads in seconds. If the data folder is inside OneDrive or another synced folder, loading can be much slower; a plain local folder is better.

### Notebook pitfalls we ran into

- **Run cells in order.** Re-running only the `meta` loading cell after the drop step makes `meta` (21,837 rows) and `X` (21,430 rows) mismatch. When in doubt: *Kernel → Restart & Run All*.
- **Never shuffle or filter `X` or `meta` alone.** Row *i* of one must stay row *i* of the other.
- **Run the normalization and transpose steps once.** The transpose step has a guard so a second run does not flip the axes back.
- If you change what gets loaded (e.g. 500 Hz or a lead subset), use a new cache file name or delete `X100.npy` first.

---

## 5. Plan going forward

1. **CNN model class:** Conv1d, 12 input channels, 5 outputs; test with a fake batch.
2. **Training loop:** multi-label loss + macro AUC, with an optional lead-dropout switch.
3. **Train model A:** dropout off; save the weights.
4. **Masked evaluation:** A on validation, 12 leads vs. 3 leads.
5. **Train model B:** same code, dropout on; save the weights.
6. **Compare on validation:** pick the dropout rate here.
7. **Final test runs:** A and B on 12 leads, on I/II/V2, and on all 220 three-lead sets.
8. **Analysis and plots:** which leads matter, which diagnoses suffer.

### Design decisions

- **Model:** start with a simple 1D CNN (reliable, strong on PTB-XL, good for debugging the pipeline). A Transformer or CNN+Transformer hybrid may follow as a comparison. A Transformer that treats each lead as a token could handle varying lead counts natively.
- **Loss / outputs:** `BCEWithLogitsLoss` (independent yes/no per class), sigmoid for probabilities. Sanity check: an untrained model's loss should be ≈ ln 2 ≈ 0.69.
- **Main metric:** macro AUC over the 5 superclasses (threshold-free, the PTB-XL standard; the benchmark paper's best models reach roughly 0.93). Also per-class AUC.
- **Fair comparison:** A and B share architecture, learning rate, batch size, epochs and seed. Only lead dropout differs.
- **Best-epoch selection:** track validation macro AUC on 12 leads and on I/II/V2 each epoch; use the same rule for A and B (e.g. the average of both).
- **Lead dropout (model B):** during training, randomly zero leads per record (each record keeps at least one); show the full 12 leads part of the time. Rate to be chosen on validation.
- **Masked evaluation:** keep the chosen leads, zero the rest. All 220 three-lead combinations are cheap to evaluate.
- **Save weights** after training so evaluations never require retraining.
- Optional: repeat A and B with 2–3 seeds to show the difference is not luck; train a model on I/II/V2 only as an upper reference.

### Related work worth reading

- Wagner et al. (2020), *PTB-XL, a large publicly available ECG dataset*, Scientific Data.
- Strodthoff et al. (2020), *Deep learning for ECG analysis: benchmarks and insights from PTB-XL* (reference AUC numbers).
- PhysioNet/CinC Challenge 2021, *"Will Two Do?"*: ECG classification from 12-, 6-, 4-, 3- and 2-lead subsets.

---

## 6. History

We first prototyped on the Kaggle *ECG Heartbeat Categorization* CSVs (MIT-BIH, single beats, 187 samples, single lead). We switched to PTB-XL because it has full 12-lead recordings, patient IDs and official patient-safe splits, which the reduced-lead question requires.