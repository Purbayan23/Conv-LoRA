# ConvLoRA Segmentation: Google Colab Run Guide

This guide describes how to run the ConvLoRA medical image segmentation project in Google Colab. It covers dataset preparation, source training, Early Segmentation Head (ESH) training, ConvLoRA target adaptation, validation, evaluation, inference, and troubleshooting.

Replace placeholders such as `<PROJECT_ROOT>`, `<DRIVE_DATASET_ROOT>`, and `<OUTPUT_ROOT>` with paths appropriate for the current Colab session and Google Drive layout.

## Project Overview

The project uses:

- Python and PyTorch.
- Plain YAML configuration loaded with PyYAML and converted to Python dataclasses.
- Deterministic seed `42`.
- RGB images with one binary segmentation mask channel.
- Fixed input resolution `256 x 256`.
- The reference-style residual `UNet2D` for ConvLoRA source and target experiments.
- The independent Ronneberger-style U-Net for the original baseline script.
- Separate source training, ESH training, target adaptation, and target evaluation stages.

## Official ISIC Sources

Use only the official ISIC Challenge pages for downloading the datasets:

- [ISIC Challenge datasets](https://challenge.isic-archive.com/data/)
- [ISIC 2016 Challenge](https://challenge.isic-archive.com/landing/2016/)
- [ISIC 2016 Task 1 lesion segmentation](https://challenge.isic-archive.com/landing/2016/37/)
- [ISIC 2017 Challenge](https://challenge.isic-archive.com/landing/2017/)

The ISIC 2016 segmentation data required by this project are:

- 900 training JPEG images.
- 900 corresponding binary PNG ground-truth masks.
- Optional official test images and masks for the separate test evaluation path.

The ISIC 2017 target data required by the adaptation workflow are the image and mask directories referenced by the frozen project manifests. Download the relevant official training/ground-truth data from the ISIC 2017 Challenge page. The project manifests define the adaptation, consistency, and target-evaluation membership.

## Workflow Used In This Project

The detailed options below are retained for portability, but the following sequence records the workflow used for the project.

### Step 1: Obtain and extract the dataset archives

**Options:**

- Download the official ZIP archives directly inside Colab.
- Download the official ZIP archives on a local workstation, then transfer them to Drive.
- Use an existing Drive-backed dataset staging area if its contents have already been verified.

**Workflow used here:** download the required official ISIC ZIP archives locally, extract them locally, verify the expected image/mask directory names and pair counts, and prepare the complete directory tree before uploading it.

### Step 2: Upload the extracted dataset to Google Drive

**Options:**

- Upload the ZIP archives and extract them in Drive.
- Upload the already extracted directories.
- Use a Drive synchronization tool when the dataset is large.

**Workflow used here:** upload the locally extracted ISIC directories to a Drive-backed dataset root while preserving the project directory structure.

### Step 3: Clone the project repository in Colab

**Options:**

- Clone the approved GitHub repository URL.
- Upload a repository archive through the Colab file browser.
- Copy a prepared project directory into the runtime.

**Workflow used here:** clone the project repository from GitHub into the Colab workspace and use that directory as `<PROJECT_ROOT>`.

### Step 4: Mount Google Drive

**Options:**

- Mount Drive for persistent dataset and checkpoint storage.
- Run entirely from Colab-local storage for a temporary session.
- Mount Drive only for dataset input or only for experiment outputs.

**Workflow used here:** mount Drive so the uploaded dataset is available and experiment outputs can be persisted.

### Step 5: Copy the dataset into the Colab runtime

**Options:**

- Read the dataset directly from Drive.
- Copy the extracted directories to the temporary Colab filesystem.
- Copy only the source or target dataset needed for the current phase.

**Workflow used here:** copy the required ISIC 2016 and ISIC 2017 directories from Drive into the Colab-local dataset root, preserving the directory names expected by the configuration.

### Step 6: Install the project requirements

**Options:**

- Install `requirements/base.txt` for training and evaluation.
- Install `requirements/dev.txt` when running tests and development checks.
- Install both files when the same runtime will be used for experiments and tests.

**Workflow used here:** install the existing base requirements after cloning the repository, then install the existing development requirements when running the test suite.

### Step 7: Configure and verify the runtime

**Options:**

- Set paths through environment variables.
- Pass paths and hyperparameters as configuration overrides.
- Keep persistent outputs on Drive and use Colab-local data for input throughput.

**Workflow used here:** set the dataset and output environment variables, select the Colab runtime configuration, verify imports and directory structure, and then run the source, ESH, adaptation, and evaluation phases.

## Colab Preparation

### 1. Start a runtime

Use a GPU runtime when possible:

1. Open **Runtime -> Change runtime type**.
2. Select a GPU accelerator.
3. Keep the runtime deterministic for reproduction. GPU availability changes execution speed but does not change the configured experiment protocol.

Check the runtime:

```python
import sys
import torch

print(sys.version)
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 2. Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

Store downloaded archives and persistent experiment outputs under a Drive-backed storage location. Set the dataset staging root through the configurable variables below.

### 3. Place the repository in Colab

Use one of the following approaches:

- Upload an archive of the ConvLoRA repository through the Colab file browser and extract it under `<PROJECT_ROOT>`.
- Clone the repository URL into `<PROJECT_ROOT>`.

The workflow used here cloned the approved GitHub repository URL:

```bash
export PROJECT_ROOT=/content/<PROJECT_DIRECTORY>
git clone "<GITHUB_REPOSITORY_URL>" "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
```

Example placeholder layout:

```bash
export PROJECT_ROOT=/content/<PROJECT_DIRECTORY>
cd "$PROJECT_ROOT"
```

The project root must contain `src/medseg`, `configs`, `scripts`, `tests`, `splits`, and `pyproject.toml`.

### 4. Install project dependencies

Install the repository dependencies in the current runtime:

```bash
cd "$PROJECT_ROOT"
python -m pip install -q -r requirements/base.txt
```

Install development dependencies only when running the test suite:

```bash
python -m pip install -q -r requirements/dev.txt
```

The project uses PyYAML, PyTorch, torchvision, NumPy, Pillow, matplotlib, and tqdm. Configuration is loaded with PyYAML and Python dataclasses.

Confirm imports:

```bash
python - <<'PY'
import torch
import yaml

print("PyTorch:", torch.__version__)
print("PyYAML:", yaml.__version__)
PY
```

## Dataset Staging

### Storage terminology

Use these terms when documenting a run:

- **Dataset root**: the environment-specific absolute directory containing the extracted ISIC dataset folders.
- **Drive dataset root**: the persistent Google Drive staging location.
- **Colab dataset root**: an optional local-runtime copy used to improve input throughput.
- **Output root**: the persistent or temporary directory where experiment runs, checkpoints, manifests, logs, and figures are stored.

These values are configurable for each runtime.

### ISIC 2016 directory layout

The configured ISIC 2016 dataset root must contain:

```text
<ISIC2016_DATASET_ROOT>/
├── ISBI2016_ISIC_Part1_Training_Data/
│   ├── ISIC_XXXXXXX.jpg
│   └── ...
├── ISBI2016_ISIC_Part1_Training_GroundTruth/
│   ├── ISIC_XXXXXXX_Segmentation.png
│   └── ...
├── ISBI2016_ISIC_Part1_Test_Data/                  # optional until test evaluation
│   └── ...
├── ISBI2016_ISIC_Part1_Test_GroundTruth/           # optional until test evaluation
│   └── ...
└── splits/                                          # generated if absent
    ├── train.csv
    └── val.csv
```

Images and masks are paired by the common `ISIC_XXXXXXX` identifier. The mask suffix `_Segmentation` is removed during pairing. Training masks are read as grayscale values `0` and `255`, then converted to binary tensor values `0` and `1`.

The source adapter creates a deterministic split from the 900 official training pairs:

- Seed: `42`.
- Train/validation ratio: `0.8`.
- Training samples: `720`.
- Validation samples: `180`.
- Train/validation intersection: `0`.
- Union: all `900` valid pairs.

Existing `train.csv` and `val.csv` manifests are reused. If they are absent, the adapter creates them under the configured ISIC 2016 dataset root. The generated split uses the configured seed and ratio.

### ISIC 2017 directory layout

The target image and mask directories can be located anywhere, provided the corresponding configuration values point to them:

```text
<ISIC2017_DATASET_ROOT>/
├── ISIC-2017_Training_Data/
│   └── ... image files ...
└── ISIC-2017_Training_Part1_GroundTruth/
    └── ... mask files ...
```

The repository contains frozen manifests:

```text
<PROJECT_ROOT>/splits/
├── isic2017_duplicate_audit.csv
├── isic2017_target_adaptation.csv
└── isic2017_target_eval.csv
```

Their roles are fixed:

- `isic2017_target_adaptation.csv`: 1003 target images used for the deterministic adaptation/consistency split.
- Adaptation subset: 803 images.
- Consistency subset: 200 disjoint images used for adaptation checkpoint selection.
- `isic2017_target_eval.csv`: separate 251-image labeled evaluation set.

The adaptation dataset is image-only. Target masks are opened by the held-out evaluation command.

### Upload archives to Drive

The available options are to upload the official ZIP archives and extract them in Drive, or to upload directories that were extracted locally. The workflow used here downloaded the official archives locally, extracted them locally, verified the directory names, and then uploaded the extracted directories to `<DRIVE_DATASET_ROOT>`.

Extracted data must preserve the configuration contract exactly. Avoid an accidental extra nested directory level.

For example, after extraction the configured root should contain `ISBI2016_ISIC_Part1_Training_Data` directly, not a second copy of that same directory name below it.

### Copy data from Drive to the Colab runtime

Copying the dataset to the local Colab filesystem is optional but often improves throughput. The copy is temporary and is lost when the runtime is reset.

For the copy cell below, organize the Drive staging root with `isic2016/` and `isic2017/` subdirectories containing the dataset-specific folders described above.

```python
from pathlib import Path
import shutil

DRIVE_DATASET_ROOT = Path("/content/drive/MyDrive/<DRIVE_DATASET_DIRECTORY>")
COLAB_DATASET_ROOT = Path("/content/<COLAB_DATASET_DIRECTORY>")

folders = [
    "isic2016/ISBI2016_ISIC_Part1_Training_Data",
    "isic2016/ISBI2016_ISIC_Part1_Training_GroundTruth",
    "isic2016/ISBI2016_ISIC_Part1_Test_Data",
    "isic2016/ISBI2016_ISIC_Part1_Test_GroundTruth",
    "isic2017/ISIC-2017_Training_Data",
    "isic2017/ISIC-2017_Training_Part1_GroundTruth",
]

for folder in folders:
    source = DRIVE_DATASET_ROOT / folder
    destination = COLAB_DATASET_ROOT / folder
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
```

Copy the ISIC 2017 image and mask directories similarly if target adaptation or target evaluation will be run in the current session.

## Runtime Configuration

Set environment variables before launching scripts. These values are runtime inputs, not hard-coded project behavior:

```python
import os

os.environ["MEDSEG_DATA_ROOT"] = "/content/<COLAB_DATASET_DIRECTORY>/isic2016"
os.environ["ISIC2017_IMAGES_DIR"] = "/content/<COLAB_DATASET_DIRECTORY>/isic2017/ISIC-2017_Training_Data"
os.environ["ISIC2017_MASKS_DIR"] = "/content/<COLAB_DATASET_DIRECTORY>/isic2017/ISIC-2017_Training_Part1_GroundTruth"
os.environ["MEDSEG_OUTPUT_ROOT"] = "/content/drive/MyDrive/<DRIVE_OUTPUT_DIRECTORY>"
```

The relevant configuration fields are:

| Purpose | Configuration field | Environment variable or override |
| --- | --- | --- |
| ISIC 2016 dataset root | `dataset.root` | `MEDSEG_DATA_ROOT` |
| ISIC 2017 image directory | `stage2.target_images_dir` | `ISIC2017_IMAGES_DIR` |
| ISIC 2017 mask directory | `stage2.target_masks_dir` | `ISIC2017_MASKS_DIR` |
| Persistent run output | `experiment.output_root` | `MEDSEG_OUTPUT_ROOT` |
| Random seed | `seed` | `seed=42` |
| Input height/width | `preprocessing.size.height/width` | `256`, `256` |
| Metric threshold | `metrics.threshold` | `0.5` |

The loader supports ordinary `key=value` overrides. Configuration groups can be selected with overrides such as `model=unet2d_source` or `stage2=convlora_3x3_only_isic2017`. Nested values can be overridden with dot notation, such as `stage2.source_checkpoint=<CHECKPOINT_PATH>`.

## Preprocessing And Augmentation

The default preprocessing contract is:

- Image color order: RGB.
- Image tensor layout: `CHW`.
- Mask tensor layout: `1HW`.
- Image dtype: `float32`.
- Mask dtype: `float32`.
- Input range: `[0, 1]` after uint8 conversion.
- Spatial size: `256 x 256`.
- Image resize interpolation: bilinear.
- Mask resize interpolation: nearest-neighbor.
- Image normalization: disabled by default.
- Model output: one-channel logits with shape `[B, 1, H, W]`.
- Binary metric threshold: `0.5` after sigmoid conversion.

Training uses paired spatial augmentation so the image and mask remain aligned:

- Random horizontal flip.
- Random vertical flip.
- Random 90-degree rotation.

Evaluation transforms are deterministic and do not apply random augmentation. Geometric transforms must be shared by image and mask, with nearest-neighbor interpolation for masks.

## Reproducibility Conditions

Use seed `42` for every phase unless a deliberately separate experiment is being defined. The project seed helper configures Python, NumPy, PyTorch, and CUDA randomness and enables deterministic settings when configured.

Recommended Colab override:

```bash
python scripts/train_stage2_source.py runtime=colab seed=42
```

The Colab runtime profile sets deterministic execution, automatic device selection, and the Colab worker/pinning defaults. For reproducibility, record the Python, PyTorch, CUDA, GPU, repository commit, configuration overrides, and output-root identity.

## Experiment Phases

All commands below are run from `<PROJECT_ROOT>`:

```bash
cd "$PROJECT_ROOT"
```

Use placeholder checkpoint variables rather than embedding environment-specific absolute paths:

```bash
export SOURCE_CHECKPOINT="<SOURCE_CHECKPOINT_PATH>"
export ESH_CHECKPOINT="<ESH_CHECKPOINT_PATH>"
export ADAPTATION_CHECKPOINT="<ADAPTATION_CHECKPOINT_PATH>"
export BASELINE_CHECKPOINT="<BASELINE_CHECKPOINT_PATH>"
```

### Phase 0: Configuration and import checks

Compose the default configuration:

```bash
python -c "from medseg.config import compose_config; c=compose_config(overrides=['runtime=colab', 'seed=42']); print(c.seed, c.preprocessing.size.height, c.preprocessing.size.width)"
```

Confirm the package import path:

```bash
python -c "from medseg.models.architectures.unet import UNetRonneberger2015; from medseg.models.architectures.unet2d import UNet2D; from medseg.models.extensions import ConvLoRA; print('imports ok')"
```

### Phase 1: Pure Ronneberger U-Net baseline

The default `scripts/train.py` path uses the independent original-style U-Net configuration. It is useful for the baseline comparison but is not the ConvLoRA source-backbone training command.

Run the configured baseline:

```bash
python scripts/train.py \
  runtime=colab \
  seed=42 \
  experiment=baseline_isic2016_unet
```

The baseline configuration controls its model, loss, optimizer, scheduler, batch size, epoch count, and output root.

Validate a saved baseline checkpoint:

```bash
python scripts/validate.py \
  runtime=colab \
  checkpoint.load_path="$BASELINE_CHECKPOINT" \
  split=val
```

Evaluate the official ISIC 2016 test split only after its image and mask directories have been extracted and configured:

```bash
python scripts/validate.py \
  runtime=colab \
  checkpoint.load_path="$BASELINE_CHECKPOINT" \
  split=test
```

The default split is `val`. The test split must remain separate from training and checkpoint selection.

### Phase 2: ConvLoRA source training

This is the source model used by the Stage 2 ConvLoRA adaptation workflow. It trains the residual `UNet2D` from scratch on ISIC 2016 with ConvLoRA disabled.

The configured source protocol is:

- 50 epochs.
- Batch size `32`.
- Adam optimizer.
- Learning rate `0.001`.
- BCE reference loss.
- No scheduler.
- Seed `42`.
- Best checkpoint selected by source validation Dice.
- Source train/validation split only.
- No ISIC 2017 target data or target masks.

Run source training:

```bash
python scripts/train_stage2_source.py \
  runtime=colab \
  seed=42 \
  dataset.root="$MEDSEG_DATA_ROOT" \
  experiment.output_root="$MEDSEG_OUTPUT_ROOT"
```

The script prints the resolved source checkpoint path and history path. Record both for the ESH and adaptation phases.

### Phase 3: ESH training

ESH training uses the frozen source `UNet2D` checkpoint and labeled ISIC 2016 train/validation data. The ESH consumes level-3 features and is trained independently.

The configured ESH protocol is:

- 20 epochs.
- Batch size `32`.
- Adam optimizer.
- Learning rate `0.001`.
- No scheduler.
- Source U-Net frozen.
- Best checkpoint selected by validation Dice.

Run ESH training:

```bash
python scripts/train_stage2_esh.py \
  runtime=colab \
  seed=42 \
  dataset.root="$MEDSEG_DATA_ROOT" \
  stage2.source_checkpoint="$SOURCE_CHECKPOINT" \
  experiment.output_root="$MEDSEG_OUTPUT_ROOT"
```

The script requires `stage2.source_checkpoint` and prints the resulting ESH checkpoint path.

### Phase 4: ConvLoRA target adaptation

Target adaptation uses only the image portion of the ISIC 2017 target adaptation manifest. It does not open target masks. The frozen ESH supplies the adaptation-side segmentation signal through the configured pseudo-label objective.

The historical full-encoder ConvLoRA protocol is:

- Insertion scope: `[init_path, down1, down2, down3]`.
- Rank: `2`.
- Alpha: `2`.
- Adapted layers: all selected convolutions in the historical scope.
- Decoder and output path: no adapters.
- BN running statistics: controlled by the selected configuration.
- Base model parameters: frozen.
- ESH: frozen.
- Adaptation batch size: `32`.
- Adaptation learning rate: `0.0001`.
- Adaptation epochs: `5`.
- Optimizer: Adam.
- Scheduler: none.
- Consistency fraction: `0.2`.
- Consistency split seed: `42`.
- Target labels allowed for training: `false`.

Run the historical full-encoder configuration:

```bash
python scripts/adapt_stage2.py \
  runtime=colab \
  seed=42 \
  stage2=convlora_isic2016 \
  model=unet2d_convlora \
  experiment=stage2_convlora_isic2016 \
  stage2.source_checkpoint="$SOURCE_CHECKPOINT" \
  stage2.esh_checkpoint="$ESH_CHECKPOINT" \
  experiment.output_root="$MEDSEG_OUTPUT_ROOT"
```

Run the preferred 3x3-only frozen-BN ConvLoRA ablation:

```bash
python scripts/adapt_stage2.py \
  runtime=colab \
  seed=42 \
  stage2=convlora_3x3_only_isic2017 \
  model=unet2d_convlora \
  experiment=convlora_3x3_only_isic2017 \
  stage2.source_checkpoint="$SOURCE_CHECKPOINT" \
  stage2.esh_checkpoint="$ESH_CHECKPOINT" \
  experiment.output_root="$MEDSEG_OUTPUT_ROOT"
```

The 3x3-only configuration excludes 2x2 stride-2 downsampling convolutions and leaves decoder/output convolutions unchanged. It uses frozen BN running statistics and the same 803/200 adaptation-consistency protocol.

The adaptation script creates a run directory containing a manifest and checkpoint artifacts. The selected checkpoint is based on consistency Dice, not on the held-out target evaluation labels.

### Phase 5: Source-only target evaluation

The source checkpoint contains the base `UNet2D` state, not ConvLoRA adapter parameters. Use the following notebook cell to evaluate it without attaching an adapter:

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from medseg.config import compose_config
from medseg.data.augmentations import build_transforms
from medseg.data.dataloaders.collate import segmentation_collate
from medseg.data.datasets import ISIC2017ManifestDataset
from medseg.losses import build_loss
from medseg.training.stage2 import build_base_unet, load_model_state
from medseg.utils import detect_device
from medseg.validation import Evaluator

source_checkpoint = Path("<SOURCE_CHECKPOINT_PATH>")
config = compose_config(
    overrides=[
        "runtime=colab",
        "seed=42",
        "stage2=convlora_3x3_only_isic2017",
        "model=unet2d_convlora",
        "loss=bce_reference",
    ]
)
device = torch.device(detect_device(config.runtime.device))
dataset = ISIC2017ManifestDataset(
    config=config,
    manifest_path=config.stage2.target_eval_manifest,
    images_dir=config.stage2.target_images_dir,
    masks_dir=config.stage2.target_masks_dir,
    transform=build_transforms(config, stage="eval"),
    include_mask=True,
)
loader = DataLoader(
    dataset,
    batch_size=config.stage2.adaptation_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=config.runtime.num_workers or 0,
    pin_memory=bool(config.runtime.pin_memory),
    collate_fn=segmentation_collate,
)
model = build_base_unet(config).to(device)
load_model_state(source_checkpoint, model, device=device)
results = Evaluator.from_config(config).evaluate(
    model=model,
    dataloader=loader,
    loss_fn=build_loss(config),
    device=device,
)
print(results)
```

This evaluation opens target masks only for reporting. It does not modify the source checkpoint or use target labels for adaptation.

### Phase 6: ConvLoRA target evaluation

Evaluate a saved ConvLoRA adaptation checkpoint:

```bash
python scripts/evaluate_stage2.py \
  runtime=colab \
  seed=42 \
  stage2=convlora_3x3_only_isic2017 \
  model=unet2d_convlora \
  stage2.source_checkpoint="$SOURCE_CHECKPOINT" \
  stage2.adaptation_checkpoint="$ADAPTATION_CHECKPOINT" \
  experiment.output_root="$MEDSEG_OUTPUT_ROOT"
```

The evaluator reports loss, Dice, IoU, precision, and recall using the configured threshold `0.5`. It uses the separate labeled target-evaluation manifest and does not change adaptation or checkpoint selection.

### Phase 7: Checkpoint validation and inference

For the Pure U-Net validation/test scripts, provide the checkpoint with `checkpoint.load_path` and select the appropriate model/configuration group. Keep `split=val` for validation and use `split=test` only for the official ISIC 2016 test set.

```bash
python scripts/validate.py \
  runtime=colab \
  model=unet_ronneberger2015 \
  checkpoint.load_path="$BASELINE_CHECKPOINT" \
  split=val
```

```bash
python scripts/infer.py \
  runtime=colab \
  model=unet_ronneberger2015 \
  checkpoint.load_path="$BASELINE_CHECKPOINT"
```

Extract the official ISIC 2016 test data before using `split=test`; validation data are not a substitute for the test split.

## Available Helper Functions

The following helpers are useful when debugging or building a notebook around the scripts:

| Area | Helper | Purpose |
| --- | --- | --- |
| Configuration | `compose_config` | Compose grouped YAML files and apply `key=value` overrides |
| Configuration | `load_typed_config` | Load the composed configuration into dataclasses |
| Reproducibility | `seed_everything` | Seed Python, NumPy, PyTorch, and CUDA where available |
| Dataset | `build_dataset` | Construct the configured dataset adapter for one split |
| Dataset | `ISIC2016Dataset` | Pair ISIC 2016 images and segmentation masks |
| Dataset | `ISIC2017ManifestDataset` | Read frozen target manifests with image-only or labeled-evaluation modes |
| Transforms | `build_transforms` | Build paired image/mask preprocessing and augmentation |
| Transforms | `build_image_transform` | Build image-only target preprocessing |
| Dataloaders | `build_dataloaders` | Construct requested train, validation, or test loaders |
| Models | `build_model` | Construct the model selected by configuration |
| Models | `UNet2D` | Construct the active residual source/adaptation backbone |
| Models | `UNetRonneberger2015` | Construct the independent original-style baseline |
| Losses | `build_loss` | Construct the configured BCE, Dice, or combined loss |
| Metrics | `Evaluator` | Evaluate loss and aggregated binary metrics |
| Metrics | `binary_confusion_counts` | Compute binary TP/FP/FN counts from logits and masks |
| Optimization | `build_optimizer_spec` | Resolve optimizer settings from YAML |
| Training | `Trainer` | Run model-agnostic source training and validation |
| Stage 2 | `prepare_adaptation_model` | Load a source model and attach ConvLoRA |
| Stage 2 | `prepare_frozen_esh` | Load and freeze ESH |
| Stage 2 | `split_adaptation_dataset` | Create the deterministic 803/200 split from the frozen manifest |
| Stage 2 | `adapt_model` | Run target adaptation and consistency-based checkpoint selection |
| Checkpoints | `load_model_state` | Load project-wrapped or raw PyTorch model states |
| Experiments | `create_run_directories` | Create the canonical output/checkpoint/figure structure |

## Output Structure

Set `experiment.output_root` to a persistent Drive-backed output root when checkpoints must survive runtime resets. Each run is organized by dataset, version, split definition, model, seed, experiment identity, and timestamp:

```text
<OUTPUT_ROOT>/
└── dataset=<name>/
    └── dataset_version=<version>/
        └── split_definition=<definition>/
            └── model=<model>/
                └── seed=42/
                    └── experiment=<experiment>/
                        └── <timestamp>/
                            ├── run_manifest.json
                            ├── history.json
                            ├── checkpoints/
                            │   ├── best.pt
                            │   └── latest.pt
                            ├── logs/
                            ├── predictions/
                            └── figures/
```

Use a distinct `experiment.name` or output root for each controlled ablation.

## Lightweight Verification Checks

Run the focused tests after installing development dependencies:

```bash
pytest -q tests/test_config.py
pytest -q tests/test_dataset_contract.py tests/test_transform_contract.py
pytest -q tests/test_model_builder.py tests/test_convlora_architecture.py
pytest -q tests/test_convlora_only.py tests/test_convlora_3x3_only.py
pytest -q tests/test_stage2.py tests/test_evaluator_metrics.py
pytest -q tests/test_train_smoke.py
```

Run the full suite when the runtime is sufficiently provisioned:

```bash
pytest -q
```

The smoke tests are not a substitute for a full experiment. They verify configuration composition, dataset contracts, model construction, forward/loss/backward behavior, checkpoint handling, metric aggregation, and targeted adaptation invariants.

## Common Errors And Remedies

### `ModuleNotFoundError` for `torch`, `yaml`, or `PIL`

Confirm that the Colab runtime is active and run the base dependency installation from `<PROJECT_ROOT>`. Restart the runtime only if the package installation reports that a restart is required, then rerun the import check.

### Dataset directory not found

Check the environment variables and the resolved directory tree:

```bash
echo "$MEDSEG_DATA_ROOT"
echo "$ISIC2017_IMAGES_DIR"
echo "$ISIC2017_MASKS_DIR"
find "$MEDSEG_DATA_ROOT" -maxdepth 2 -type d | sort
```

The directory names must match the YAML contract exactly. An extra nested directory level is a common extraction error.

### No ISIC image/mask pairs found

Verify that images use the `ISIC_XXXXXXX.jpg` naming scheme and masks use `ISIC_XXXXXXX_Segmentation.png`. Confirm that the image and mask identifiers match and that the mask directory contains binary PNG files.

### Missing or unexpected train/validation manifests

The ISIC 2016 adapter generates `train.csv` and `val.csv` only when they are absent. Existing ISIC 2017 manifests are frozen and should remain unchanged during ordinary runs.

### Target manifest validation error

Check that the repository `splits/` directory contains the three expected ISIC 2017 manifests and that the target image directory matches the manifest paths or fallback filename resolution. The adaptation manifest must declare that target labels are not allowed for training.

### Checkpoint path or state-dict mismatch

Confirm that the model configuration used for loading matches the checkpoint-producing phase. Source checkpoints use `model=unet2d_source`; ConvLoRA adaptation checkpoints use `model=unet2d_convlora` and the matching Stage 2 configuration. Pass explicit checkpoint overrides rather than relying on a previous notebook variable.

### CUDA out-of-memory

Reduce the configured batch size through an override, for example `stage2.adaptation_batch_size=16`, and record the change as a protocol deviation. Clear stale tensors between notebook cells and avoid retaining output graphs.

### DataLoader worker or notebook hang

Set `runtime.num_workers=0` and `runtime.pin_memory=false` for debugging. After the data path is confirmed, restore the Colab runtime settings if they improve throughput.

### Non-reproducible results

Record the repository commit, seed, resolved configuration, Python/PyTorch versions, device, dataset manifest hashes, and checkpoint paths. Ensure training uses `seed=42`, deterministic runtime settings, fixed manifests, and the same augmentation policy.

### Target-label leakage concern

Adaptation uses `ISIC2017ManifestDataset` with `include_mask=False`; evaluation uses `include_mask=True`. Checkpoint selection uses the consistency subset rather than target-evaluation labels.

## Run Checklist

Before a source run:

- Official ISIC 2016 training image and ground-truth archives were obtained from the official ISIC pages.
- The extracted directory names match the expected layout.
- `MEDSEG_DATA_ROOT` resolves to the ISIC 2016 dataset root.
- Seed is `42`.
- Dataset count is 900 before splitting.
- Train/validation counts are 720/180.
- Source model is `UNet2D` with `n_filters_init=16` for the adaptation workflow.
- ConvLoRA is disabled during source training.

Before ESH training:

- The source checkpoint exists and loads successfully.
- Source train/validation loaders are available.
- ESH checkpoint output is assigned to a persistent location.

Before target adaptation:

- The three frozen ISIC 2017 manifests are present.
- The 1003 adaptation manifest rows resolve to images.
- The 803/200 split is deterministic with seed `42`.
- The adaptation dataset is image-only.
- The source and ESH checkpoints are explicitly supplied.
- The intended ConvLoRA configuration is selected.

Before target evaluation:

- The adaptation checkpoint exists.
- The target-evaluation manifest contains 251 separate labeled samples.
- Target masks are available only for the evaluation command.
- The threshold remains `0.5` unless a separately documented diagnostic requires another value.

## Scientific Reporting Requirements

For every run, retain:

- Repository commit identifier.
- Resolved configuration and command-line overrides.
- Dataset root description and environment-variable settings.
- Dataset and manifest counts.
- Seed and deterministic settings.
- Model and adapter settings.
- Optimizer, learning rate, scheduler, batch size, and epochs.
- Source, ESH, and adaptation checkpoint identities.
- Validation and target-evaluation metrics.
- Any runtime deviations such as reduced batch size, worker changes, or CPU execution.

Validation metrics are distinct from official test performance. Adaptation checkpoint selection uses the consistency subset rather than held-out target-evaluation labels.
