# Project Audit — GTSRB Traffic Sign Recognition

**Audit date:** 2026-09-24
**Audited revision:** `2115ff2` — *"Add files via upload"* (the only commit on `main`)
**Auditor:** repository refactor initiative

This document is the factual baseline for the modernisation work. Everything below was
read out of the original artefacts in this repository; nothing is inferred from
external sources or from memory. Where a claim comes from a recorded notebook run it is
marked **(observed)**. Where a claim is a code-reading conclusion it is marked **(code)**.

---

## 1. What the original repository actually contained

| Artefact | Size | Purpose |
| --- | --- | --- |
| `COMP9444_notebook.ipynb` | 7.3 MB | The entire project: 58 cells (25 code, 33 Markdown), including stored outputs |
| `COMP9444_ppt.pptx` | 18.7 MB | Course presentation deck |
| `COMP9444_report.pdf` | 686 KB | 4-page IEEE-style group report |

Total repository size: **50 MB**, of which ~26 MB is committed binary payload at the
tip of `main`. There was **no README, no `requirements.txt`, no `pyproject.toml`, no
`.gitignore`, no LICENSE, no tests, no CI, and no source package.**

Provenance: this was a university group project for **COMP9444 Neural Networks and Deep
Learning** (University of Sydney). Authors, as listed on the report: Ashwin Sudhir
Jamgade, Ishu Rajput, Shashwat Pasari, Xiangyu Dou.

---

## 2. Current architecture

The original "architecture" is a single linear notebook. Its execution order is:

```text
Colab GPU setup (A100)
        |
        v
Seeds (SEED=43) + device select + constants duplicated in the cell namespace
        |
        v
Training transforms (64x64 GTSRB stats)  +  (224x224 ImageNet stats for ResNet)
        |
        v
torchvision.datasets.GTSRB  (download=True, root=/content/dataset)
        |
        v
EDA: class-frequency bar chart, one-sample-per-class grid
        |
        v
Stratified 80/20 split of the *training* split  ->  torch.utils.data.Subset
        |
        v
AugmentedGTSRB(Dataset): materialises an oversampled index list (floor = 800/class)
        |
        v
DataLoaders (batch 16) x 6  (64x64: train/val/test, 224x224: train/val/test)
        |
        v
Three model classes defined inline: Custom_CNNClassifier, Custom_MLPClassifier,
ResNet50Classifier
        |
        v
train_loop()  ->  best checkpoint by validation accuracy  ->  *.pth in the Colab CWD
        |
        v
test_loop()  ->  accuracy, macro/micro P/R/F1, confusion matrix, per-class gallery,
                 misclassified gallery
        |
        v
pandas.concat of the three result frames  ->  final comparison table
```

Every stage shares one global namespace. There are no function boundaries between
configuration, data, model, training and evaluation beyond the handful of `def`s
(`train_loop`, `test_loop`, `plot_confusion`, `make_loader`, `unnormalize`, and the
three visualisation helpers).

### 2.1 Model architecture actually implemented

**`Custom_CNNClassifier`** — 64×64×3 input, three blocks of
`Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → ReLU → MaxPool2 → Dropout(0.25)`
with channel widths 64 → 128 → 256 (spatial 64 → 32 → 16 → 8), then
`Flatten → Linear(256·8·8, 512) → ReLU → Dropout(0.5) → Linear(512, 43)`. **(code)**

**`Custom_MLPClassifier`** — `Flatten(3·64·64 = 12288) → Linear(12288, 2048) → BN → ReLU
→ Dropout(0.3) → Linear(2048, 1024) → BN → ReLU → Dropout(0.3) → Linear(1024, 512) → BN
→ ReLU → Dropout(0.3) → Linear(512, 256) → BN → ReLU → Dropout(0.2) → Linear(256, 43)`.
**(code)**

**`ResNet50Classifier`** — `torchvision.models.resnet50(pretrained=True)`; all parameters
frozen **except `layer4`**; `fc` replaced with `Identity`; head =
`Linear(2048, 512) → BatchNorm1d → ReLU → Dropout(0.5) → Linear(512, 43)`. **(code)**

### 2.2 Training protocol actually implemented

- Loss: `nn.CrossEntropyLoss` **(code)**
- Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)` for CNN/MLP; `lr=5e-3` for ResNet **(code)**
- Scheduler: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`, stepped with
  `val_acc * 100` **(code)**
- Mixed precision: `torch.cuda.amp.GradScaler` + `torch.cuda.amp.autocast`, guarded by
  `torch.cuda.is_available()` **(code)**
- Gradient clipping: `clip_grad_norm_(params, 1.0)` **(code)**
- Epochs: 30 for all three models **(code)**
- Checkpoint policy: save `state_dict` whenever validation accuracy improves **(code)**
- Reproducibility: `random/np/torch/cuda` seeds fixed to 43,
  `cudnn.deterministic=True`, `cudnn.benchmark=False` **(code)**

### 2.3 Data handling actually implemented

- Source: `torchvision.datasets.GTSRB(root='/content/dataset', download=True)` **(code)**
- Split sizes recorded in the run: **train split = 26,640 images**, **test split = 12,630
  images** **(observed)**
- Class counts in the training split: **min 150, max 1500** **(observed)**
- Stratified 80/20 split of the training split: **21,312 train / 5,328 validation**
  **(observed)**
- Materialised oversampling to a per-class floor of 800: training set grows to
  **36,944 images, min 800 / max 1200 per class** **(observed)**
- Train transforms (64×64): `Resize(64)`, `RandomRotation(15)`, `RandomAffine(0,
  translate=(0.1, 0.1))`, `ColorJitter(brightness=0.2, contrast=0.2)`, `ToTensor`,
  `Normalize(GTSRB stats)` **(code)**
- Normalisation statistics stored as constants:
  `mean=[0.3403, 0.3121, 0.3214]`, `std=[0.2724, 0.2608, 0.2669]` for the 64×64 branch and
  ImageNet statistics for the ResNet branch **(code)**
- Validation/test transforms: `Resize`, `ToTensor`, `Normalize` only **(code)**
- Leakage control: split is taken **before** oversampling and augmentation, and
  oversampling is applied to the training index list only **(code, and confirmed by the
  recorded sizes)**

#### Verification performed during the audit

The refactored data pipeline was run against the real GTSRB download and reproduces the
original figures exactly:

| Quantity | Original notebook | Refactored pipeline |
| --- | --- | --- |
| Training split size | 26,640 | 26,640 |
| Classes present | 43 | 43 / 43 |
| Per-class min / max | 150 / 1500 | 150 / 1500 |
| Imbalance ratio | 10.0x | 10.00x |
| Stratified 80/20 split | 21,312 / 5,328 | 21,312 / 5,328 |

The normalisation constants were also re-measured from the data. On the 64x64-resized
training split (n = 4000, seed 43):

| Estimator | Mean | Std |
| --- | --- | --- |
| Pixel-pooled (statistically correct for normalisation) | 0.3285, 0.3006, 0.3092 | 0.2706, 0.2612, 0.2666 |
| Mean of per-image means | 0.3383, 0.3104, 0.3199 | 0.2006, 0.1870, 0.1871 |
| **Configured in the original project** | **0.3403, 0.3121, 0.3214** | **0.2724, 0.2608, 0.2669** |
| Native resolution (no resize) | 0.3741, 0.3460, 0.3550 | 0.2995, 0.2948, 0.3006 |

The configured mean matches the *mean of per-image means* to within 0.002 and the
configured std matches the *pooled* estimator to within 0.002, so the original constants
were derived with that mixed estimator at 64x64. The discrepancy against the pooled
estimator is at most 0.012 — about 4% of one standard deviation — which is immaterial to
training and is not worth breaking comparability with the recorded run for. It is
documented rather than silently "fixed". **(measured)**

### 2.4 Results actually recorded in the notebook

All values below are the stored outputs of the original Colab A100 run, captured in the
notebook as stream output (the metric tables) — not reconstructed or re-estimated.

Best validation accuracy during training: **CNN epoch 28**, **MLP epoch 30**,
**ResNet-50 epoch 29** **(observed)**.

Test-set metrics (GTSRB `test` split, 12,630 images) **(observed)**:

| Metric | Custom CNN | MLP | ResNet-50 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.988440 | 0.658432 | 0.969675 |
| Precision (macro) | 0.983161 | 0.649011 | 0.951391 |
| Recall (macro) | 0.979886 | 0.652216 | 0.967075 |
| F1 (macro) | 0.979838 | 0.635014 | 0.957855 |
| Precision (micro) | 0.988440 | 0.658432 | 0.969675 |
| Recall (micro) | 0.988440 | 0.658432 | 0.969675 |
| F1 (micro) | 0.988440 | 0.658432 | 0.969675 |

The notebook also stores, as cell outputs, the confusion matrices, per-class galleries,
misclassified-example grids and training curves for all three models. These are the
**real** evaluation artefacts of this project and they are preserved in `docs/images/`
by the refactor (see §5).

### 2.5 Error structure visible in the stored CNN confusion matrix

The stored row-normalised confusion matrix was digitised by inverting its colorbar
mapping, so the numbers below are derived from the real figure rather than eyeballed:

| True class | Predicted as | Share of true class |
| --- | --- | ---: |
| 27 Pedestrians | 23 Slippery road | 29.5% |
| 27 Pedestrians | 30 Beware of ice/snow | 16.3% |
| 6 End of speed limit 80 | 33 Turn right ahead | 4.9% |
| 12 Priority road | 15 No vehicles | 3.6% |
| 26 Traffic signals | 30 Beware of ice/snow | 2.4% |

Lowest-recall classes under the CNN: **27 Pedestrians (54.2%)**, 6 End of speed limit 80
(91.7%), 12 Priority road (96.4%), 22 Bumpy road (97.6%), 26 Traffic signals (97.6%).
Every other class is at or above 98.7%. The macro-F1 (0.9798) sits ~0.9 points below
accuracy (0.9884), and that gap is essentially entirely attributable to class 27.

---

## 3. Current problems

Ordered by how much they hurt the repository as an engineering artefact.

### 3.1 Presentation and credibility
1. **No README.** A visitor sees three opaque binaries. There is no statement of what the
   project does, how to run it, or what the results are.
2. **26 MB of binary payload at the tip** (7.3 MB notebook + 18.7 MB deck) makes clone
   slow and buries the actual work.
3. **No LICENSE**, so the code is technically all-rights-reserved.
4. **Course framing.** The repository is named and presented as a university submission,
   not as a reusable system.

### 3.2 Reproducibility
5. **No dependency declaration at all.** The notebook records the Colab Python version
   (3.11.5) but no package versions. `torch.utils.data`, `torchvision`, `scikit-learn`,
   `pandas` and `matplotlib` versions are unknown, and several of the APIs used are
   already deprecated (§3.4).
6. **No persisted split manifest.** The 80/20 split is recomputed from a seed inside the
   notebook. It is reproducible in principle, but there is no artefact anyone can audit
   to confirm that two runs used the same partition.
7. **No DataLoader seeding.** The loaders are constructed without a `generator`, so the
   shuffle order is not tied to the seed. `num_workers=4` worker RNG is also unseeded.
8. **Hardcoded Colab paths.** `/content/dataset/` is baked into the data loading; running
   anywhere else requires editing code.
9. **Configuration is scattered.** Normalisation constants are repeated verbatim in the
   transform definitions and again in the evaluation calls; image sizes, batch size,
   epoch counts and learning rates are literals at their point of use.

### 3.3 Engineering
10. **Not importable.** Models, training, evaluation and visualisation only exist inside
    notebook cells. Nothing can be reused by a service, a script or a test.
11. **No tests.** There is no automated check of preprocessing, label mapping, model
    output shape, or metric computation.
12. **No CI.** Nothing verifies that the notebook still runs, on any change.
13. **No inference path.** Producing a single prediction requires re-running the notebook
    up to the model definition and checkpoint load.
14. **No experiment tracking.** The only record of hyperparameters and metrics is
    whichever cell outputs happen to be saved in the notebook blob. Comparing runs means
    reading notebook diffs.
15. **Evaluation artefacts are not persisted** as files. The confusion matrices and error
    galleries exist only as base64 inside the `.ipynb`.
16. **In-memory oversampling.** `AugmentedGTSRB.__init__` builds a 36,944-element index
    list eagerly, and `np.random.choice` draws from the global NumPy RNG, so the
    resampling is not controlled by a local generator.

### 3.4 Code health
17. **Deprecated APIs in use** (each produced a warning in the recorded run):
    `torch.cuda.amp.GradScaler(...)` and `torch.cuda.amp.autocast(...)` — deprecated in
    favour of `torch.amp.GradScaler('cuda', ...)` / `torch.amp.autocast('cuda', ...)`;
    `ReduceLROnPlateau(verbose=True)` — deprecated; `resnet50(pretrained=True)` —
    deprecated in favour of the `weights=` enum API.
18. **CUDA-only acceleration.** AMP is gated on `torch.cuda.is_available()` and the device
    is `cuda` or `cpu`, so Apple-Silicon MPS is never used and mixed precision silently
    does nothing on non-CUDA hardware.
19. **Duplicated loader plumbing.** `make_loader` is called six times with hand-built
    dataset objects, and the same normalisation constants are re-passed to
    `test_loop` as literal tuples that must stay in sync with the transforms.

### 3.5 Documentation drift between the report and the code
These are real, checkable mismatches between `COMP9444_report.pdf` and
`COMP9444_notebook.ipynb`. They matter because the report is the project's written
record and a reader will trust it.

20. The report says the CNN classifier head uses an **adaptive average pooling layer
    producing a 6×6 map**. The code uses three `MaxPool2d(2)` layers producing 8×8 and
    then `Flatten`. **(code vs report)**
21. The report describes the MLP as having **two hidden layers of 512 and 256 neurons**.
    The code implements **four hidden layers** (2048, 1024, 512, 256). **(code vs report)**
22. The report says ResNet-50 **`layer3` and `layer4` were unfrozen**. The code unfreezes
    **only `layer4`**. **(code vs report)**

### 3.6 Methodology notes (not defects, but must be stated honestly)
23. **The reported numbers are not a full-GTSRB benchmark.** `torchvision`'s `GTSRB`
    `train` split contains 26,640 images, whereas the official GTSRB training partition
    contains 39,209. The original project therefore trained on 26,640 images and
    evaluated on the official 12,630-image test split. Any comparison against published
    GTSRB results must carry that caveat.
24. **The reported results were produced once**, on a Colab A100, with no repeats and no
    confidence intervals. They are a single honest observation, not a benchmark with
    variance.
25. **Model selection used validation accuracy only.** Accuracy is not the most sensitive
    criterion for the long-tailed classes that actually fail (class 27), so the
    checkpoint chosen was not necessarily the best on macro-F1.

---

## 4. Proposed architecture

The target is a layered, importable package with one responsibility per module, a single
validated configuration object, and separately runnable train / evaluate / predict /
serve entry points.

```text
configs/*.yaml  ─┐
                 ├─►  gtsrb.config        (Pydantic v2 validation + dotlist overrides)
env / .env     ─┘            │
                             ▼
raw GTSRB ──► gtsrb.data ──► splits manifest ──► DataLoaders
                  │  (validation, preprocessing, augmentation, class-balanced sampling)
                  ▼
            gtsrb.models (factory: compact_cnn | mlp | resnet50)
                  │
                  ▼
            gtsrb.training (trainer, losses, metrics, callbacks)
                  │            │
                  │            └──► gtsrb.tracking (MLflow: params, metrics, artefacts,
                  │                                  registered model)
                  ▼
            checkpoints/*.pt  (weights + config + class labels + metrics)
                  │
      ┌───────────┴────────────┐
      ▼                        ▼
gtsrb.evaluation        gtsrb.inference
(metrics, confusion,    (predictor: single + batch)
 error analysis,               │
 saved artefacts)              ▼
                        gtsrb.api (FastAPI: /predict /health /ready /metrics)
                               │
                               ▼
                        Docker image ──► client
```

Layering rule: `config` depends on nothing; `data`, `models` and `tracking` depend on
`config`; `training` and `evaluation` depend on `data` + `models`; `inference` depends on
`models`; `api` depends on `inference`. No module reaches back up a layer, and no module
reads a global.

---

## 5. Migration plan

| Original artefact | Disposition | Rationale |
| --- | --- | --- |
| `COMP9444_notebook.ipynb` | **Moved to `legacy/` (git-ignored)**, figures extracted to `docs/images/` | The code is fully refactored into `src/gtsrb/`; the 7.3 MB blob should not sit at the repo tip. Git history still contains it, and `legacy/` keeps a working copy locally. |
| `COMP9444_ppt.pptx` | **Moved to `legacy/` (git-ignored)** | Course presentation, not a project deliverable. 18.7 MB. |
| `COMP9444_report.pdf` | **Kept at `docs/original/`** | Primary written provenance for the reported metrics; small (686 KB) and genuinely useful. |
| `train_loop` | **Refactored** into `gtsrb.training.trainer.Trainer` | Needs checkpointing, early stopping, resume, AMP, device abstraction, structured logging, tracking hooks. |
| `test_loop` | **Refactored** into `gtsrb.evaluation.evaluate.evaluate_checkpoint` | Must return structured metrics and persist artefacts instead of only drawing them. |
| `plot_confusion`, `unnormalize`, gallery helpers | **Refactored** into `gtsrb.evaluation.visualization` | Pure functions over arrays; testable. |
| `AugmentedGTSRB` | **Replaced** by `WeightedRandomSampler` + transform pipelines | Materialised oversampling wastes memory and couples resampling to the dataset. Weighted sampling gives the same class balance with a fresh draw each epoch and a controllable generator. Behaviour change documented in `docs/DECISIONS.md`. |
| Three model classes | **Refactored** into `gtsrb.models` behind `build_model()` | Preserves the original three-way comparison while making the architecture a config value. |
| Normalisation constants | **Centralised** in `configs/data.yaml` | Currently duplicated across transforms and evaluation calls. |
| Class labels | **Centralised** in `gtsrb.config.labels` | 43 class names must exist once and be importable by evaluation and the API. |
| EDA cells | **Moved** to `notebooks/exploration.ipynb`, reduced to EDA | Notebooks are for exploration, not for the training path. |
| 80/20 stratified split | **Preserved exactly** (seed 43, `train_test_split`, stratify) and **persisted** as a JSON manifest | Same partition, now auditable. |
| Deprecated AMP / scheduler / `pretrained=` APIs | **Replaced** with `torch.amp` device-agnostic APIs and the `weights=` enum | Removes the warnings recorded in the original run and enables MPS. |

### Work breakdown (one branch and pull request per item)

1. `docs/project-audit` — this audit, repo hygiene, LICENSE, `.gitignore`
2. `feature/project-structure` — `src/` layout, `pyproject.toml`, config system, Makefile
3. `feature/data-pipeline` — dataset, splits, preprocessing, augmentation, validation
4. `feature/model-training` — model factory, trainer, losses, metrics, callbacks, scripts
5. `feature/mlflow-tracking` — experiment tracking and model registry integration
6. `feature/evaluation-error-analysis` — metrics, confusion matrix, error analysis, artefacts
7. `feature/inference` — checkpoint contract and predictor
8. `feature/inference-api` — FastAPI service
9. `feature/docker` — containerised inference
10. `feature/ci-cd` — GitHub Actions, Ruff, mypy, pre-commit
11. `docs/architecture` — README, architecture diagrams, decision records
12. `chore/final-polish` — logging, security review, dependency and test cleanup

---

## 6. Major technical decisions

Written up in full in [`DECISIONS.md`](DECISIONS.md). Summary of the ones this audit
forces:

| Decision | Choice | Why (short) |
| --- | --- | --- |
| Keep or replace the dataset? | **Keep GTSRB via torchvision** | The project's identity is GTSRB; `torchvision` handles download and parsing. The 26,640-image caveat is documented rather than hidden. |
| Primary architecture | **Keep the compact CNN as the default** | It is the best-performing model recorded in the original run (98.84% vs 96.97% for ResNet-50) and it is the deployment-friendly choice at 64×64. The MLP and ResNet-50 remain selectable so the original three-way comparison is reproducible. |
| Class imbalance strategy | **Weighted sampling instead of materialised oversampling** | Same balance, lower memory, different draw each epoch, local RNG. |
| Evaluation criterion | **Keep accuracy for parity, add macro-F1 and per-class recall** | The audit shows accuracy hides a 54% recall class; macro-F1 and per-class recall are what surface it. |
| Mixed precision | **`torch.amp` with the device type resolved at runtime** | Works on CUDA, is a no-op on CPU/MPS, and removes the deprecated calls. |
| Configuration | **Pydantic v2 models loaded from YAML with dotlist overrides** | Validation at the boundary, reproducibility (config is serialised into the checkpoint), and no scattered literals. |

---

## 7. What is explicitly *not* changing

- The problem: 43-class German traffic sign classification from cropped sign images.
- The dataset and its official test split.
- The three-model comparison: compact CNN, MLP baseline, ResNet-50 transfer learning.
- The 80/20 stratified split with seed 43.
- The leakage discipline: split first, then balance/augment the training partition only.
- The recorded results. They are reported as **observed in the original run**, with the
  exact commands needed to reproduce them, and are never re-attributed to this refactor.
