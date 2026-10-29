# Expected Dataset Layout

This repository does not store raw medical data in Git. It includes dataset adapters for ISIC 2016 source training and ISIC 2017 target adaptation/evaluation, with configuration-driven preprocessing and split handling.

## Recommended Local Layout

```text
data/
├─ isic2016/
│  ├─ ISBI2016_ISIC_Part1_Training_Data/
│  ├─ ISBI2016_ISIC_Part1_Training_GroundTruth/
│  ├─ ISBI2016_ISIC_Part1_Test_Data/
│  ├─ ISBI2016_ISIC_Part1_Test_GroundTruth/
│  └─ splits/
│     ├─ train.csv
│     ├─ val.csv
│     └─ test.csv
└─ isic2017/
   ├─ ISIC-2017_Training_Data/
   └─ ISIC-2017_Training_Part1_GroundTruth/
```

## Notes

- File names for each image and mask pair should align deterministically.
- Split manifests are the preferred source of truth even if directory names already indicate the split.
- The ISIC 2016 adapter reads existing manifests when available; otherwise it discovers image-mask pairs and creates deterministic train/validation manifests using the configured ratio and seed.
- The ISIC 2017 adapter reads the frozen manifests in `../splits/` and does not generate or edit target splits.
- Committed ISIC 2017 manifest path fields are dataset-relative filenames; the image and mask directories are supplied by configuration or environment variables.
- Masks may be stored with raw values such as `{0,255}`, but dataset adapters must convert them into the common binary contract `{0,1}`.
- The adapters expose standardized image, mask, sample-ID, and metadata fields to the trainer.
- Preprocessing and augmentation are applied through the configured transform pipeline; images and masks use separate interpolation policies where required.
- Stage 2 adaptation loads ISIC 2017 images without masks. Target masks are used only by the separate evaluation path.

## Configurable Root

The ISIC 2016 dataset root can be controlled with:

- `MEDSEG_DATA_ROOT`
- `configs/dataset/isic2016.yaml`

Stage 2 ISIC 2017 image and mask directories can be controlled with:

- `ISIC2017_IMAGES_DIR`
- `ISIC2017_MASKS_DIR`
- `configs/stage2/*.yaml`

The repository does not download datasets. Raw data and generated ISIC 2016 manifests remain local and are excluded from Git.
