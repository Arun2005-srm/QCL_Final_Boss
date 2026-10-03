# QCL Final Boss — SAM quantum continual segmentation

Config-driven RGB semantic segmentation with a **frozen original SAM ViT-B image encoder**, an eight-qubit PennyLane residual adapter, and a trainable semantic decoder. No DINO model is used. OpenEarthMap and LandCover.ai are the first dataset presets.

The executable pipeline supports training, held-out testing, byte-capped continual replay, classical adapter controls, and detailed reports. Accuracy gains and A100 throughput are **unmeasured** until real-data experiments run. This is an engineering implementation, not a completed scientific validation of the reviewed research protocol. See [implementation status](docs/IMPLEMENTATION_STATUS.md).

## A100 notebook: 20-epoch regularized experiment

Open [Run_Regularized_Project_A100.ipynb](Run_Regularized_Project_A100.ipynb) in Jupyter on the server.
It defaults to `/raid/workspace/QCL_Final_Boss`, runs 20 epochs **per region**, enables paired image/mask flips,
and lowers the decoder/classical and quantum learning rates to `0.0003` and `0.003`.
SAM feature caching is disabled for correct online augmentation, which increases runtime.
This is an exploratory configuration motivated by a training/validation gap, not a verified score improvement.

```bash
cd /raid/workspace/QCL_Final_Boss
git pull --ff-only origin main
```

Stop the old training process before starting a fresh notebook run on the same GPU. Keep its outputs for comparison.
Check the Settings cell and run the notebook from the top. See [server instructions](docs/A100_NOTEBOOK.md)
for dataset paths, exclusions, checkpoint behavior, and validation/test selection.
The original [full-budget notebook](Run_Full_Project_A100.ipynb) is retained as a baseline.

## Setup

Use Python 3.10+ in a fresh environment. On the A100 machine, install a CUDA-compatible PyTorch/torchvision pair using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
python -m pytest -q
```

Place the original SAM ViT-B checkpoint at `checkpoints/sam_vit_b_01ec64.pth`, or configure its path. Obtain it from the [official Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints). Missing weights cause an error; the pipeline never silently substitutes a random encoder. Record the installed environment with `pip freeze > environment-lock.txt` for each experiment.

Run commands from the repository root. Config inheritance paths resolve relative to the YAML file; dataset/checkpoint/cache/output paths resolve relative to the working directory. CSV image/mask paths resolve relative to that CSV.

## OpenEarthMap first

Prepare the downloaded source scenes. Expected source layout is `<region>/images/` and `<region>/labels/`.

```bash
python prepare_dataset.py --kind openearthmap --root /data/OpenEarthMap --output data_local/openearthmap --groups-csv /data/oem_source_groups.csv
python audit_dataset.py --config configs/datasets/openearthmap.yaml
python stage0.py --config configs/datasets/openearthmap.yaml --rank-check
python train.py --config configs/datasets/openearthmap.yaml
python testing.py --checkpoint outputs/openearthmap/final.pt
```

Source-group CSV columns: `id,group,domain`. `id` is the source image path relative to the dataset root, without its extension, e.g. `aachen/images/aachen_1`. Group overlapping or related source scenes together. `domain` is the geographic region or explicitly defined continual domain.

If an independent-scene audit has established that each source is independent, the preparation command accepts `--assume-independent-scenes` instead of the group CSV. This is an explicit assertion, not automatic overlap detection. Source filenames alone cannot establish geographic independence.

The OEM preset maps official raw IDs 1–8 to model IDs 0–7 and ignores unknown raw ID 0. Every mask is audited; unknown IDs fail. OEM image compilation may require obtaining its underlying xBD images separately; see the [dataset authors' instructions](https://github.com/bao18/open_earth_map#note-for-xbd-data). Use only available labelled scenes for this custom split; hidden official test labels cannot be evaluated locally.

## LandCover.ai next

Expected source layout is `images/` and `masks/`. Prepare **source orthophotos**, retaining all derived tiles under the same source group.

```bash
python prepare_dataset.py --kind landcover_ai --root /data/landcover_ai --output data_local/landcover_ai --groups-csv /data/landcover_source_groups.csv
python audit_dataset.py --config configs/datasets/landcover_ai.yaml
python train.py --config configs/datasets/landcover_ai.yaml
python testing.py --checkpoint outputs/landcover_ai/final.pt
```

The preset has five classes: background, building, woodland, water, road. This is a separate experiment with a separate classifier/PCA fit. It is not automatically appended to OEM's eight-class stream. To study cross-dataset continual learning, first define and review a common label ontology. OEM itself contains imagery from several source datasets including LandCover.ai, so independent cross-dataset evaluation also needs source-overlap checks.

LandCover.ai defaults to one domain. To study continual learning within it, provide defensible multi-scene domain groups in the source CSV. Do not treat single image tiles as independent geographic domains.

## New datasets: configuration only

Supported contract: RGB images and aligned semantic masks, binary or multiclass, with a fixed vocabulary across the stream. Masks may be integer/palette IDs or RGB colors. The current SAM model does not ingest arbitrary multispectral bands, volumes, instance IDs, or changing class vocabularies automatically. Convert these to the explicit RGB semantic contract first.

Create `samples.csv`:

```csv
id,image,mask,group,domain
tile_001,images/tile_001.png,masks/tile_001.png,source_scene_01,region_a
tile_002,images/tile_002.png,masks/tile_002.png,source_scene_01,region_a
```

Copy `configs/datasets/example.yaml`, update the manifest, class names, label mapping and output directory, then call `train.py`. Binary segmentation uses **two** classes, including background, with softmax/CE. Color masks use `color_map: {'0,0,0': 0, '255,0,0': 1}`; unmapped colors fail. Use `255` as the model ignore ID. Maximum supported vocabulary with byte-budgeted replay is 255 classes.

Folder-only pairing is also available: set `manifest: null`, configure `images`, `masks`, suffixes, and explicitly set `independent_images: true` only when images are independent. IDs use relative subpaths to avoid silently pairing duplicate basenames.

## 70/15/15 split

The requested train/validation/test ratios are **70/15/15 of independent groups within each domain**, using seeded largest-remainder rounding. Group membership is never divided. Image ratios may differ when group sizes differ. At least seven groups per domain are required to populate all three partitions. Groups spanning domains are rejected until the related domains are merged.

The split manifest records every sample, group, domain, image hash and mask hash. Duplicate image content in different groups is rejected. Test data is never used by the training loop or PCA initialization. Testing reuses the saved split and checks its hash against the checkpoint.

This user-requested split supersedes the earlier research plan's four-way split. It does **not** provide an independent diagnostic/calibration partition or independent development-domain roster. The original confirmatory plasticity gates must not be claimed from these ordinary training runs. Custom 70/15/15 results are not directly comparable to official benchmark splits.

## Training and epoch logs

```text
region_a ep[1/112] updates=18 | train acc=... loss=... | val acc=... loss=... Dice=... IoU=... mIoU=... bIoU=...
val class IoU: bareland=... | rangeland=... | ...
```

Each epoch logs full train and validation metrics to `history.json`; the console prints the requested summary and class IoUs. Validation covers all seen domains, so it shows retention as training advances. Loss is the sample-weighted mean of batch segmentation losses (CE + 0.5 Dice), excluding replay penalties from the displayed train loss. Training metrics are measured on predictions made during optimization; validation is an eval-mode pass.

By default each domain has a 2,000-update cap and a 200-epoch ceiling. The displayed epoch count is calculated from those limits and loader length; the last epoch may be partial. `run_status.json` records achieved updates. A run stopped by its epoch ceiling before 2,000 updates is not a full-budget research comparison.

```bash
python train.py --config configs/datasets/openearthmap.yaml --resume outputs/openearthmap/last.pt
```

Checkpoints include trainable model state, encoder hash, PCA, alpha, optimizer, replay, RNG states, loader generator, split hash, history and continual position. They exclude the frozen encoder weights, which must still be available. Resume is at epoch boundaries with an identical config. Default no-augmentation runs preserve the relevant loader ordering; exact replay of worker-local augmentation RNG with persistent workers is not promised. Load only trusted project checkpoints.

`final.pt` is the final stream endpoint used for the primary held-out report. `stage_XX.pt` stores each domain endpoint. `best_stage_XX.pt` is selected by that stage's all-seen validation mIoU and is an exploratory alternative, not a replacement for a fixed-update endpoint.

## Metrics and test outputs

Scores are fractions in [0,1], except loss, MCC and kappa. Undefined metrics become JSON `null`; they are not silently filled with zero.

| Output | Definition / contents |
|---|---|
| Accuracy | Correct / valid pixels |
| Dice | Macro class Dice, equal to per-class F1 before averaging |
| IoU | Micro IoU over all classes |
| mIoU | Mean class IoU from the **aggregated** confusion matrix |
| bIoU | Macro Boundary IoU, erosion width 2% of valid image diagonal, at least one pixel |
| Class metrics | IoU, Dice, precision, recall, specificity, Boundary IoU, ground-truth support |
| Additional metrics | Balanced accuracy, macro precision/recall/specificity, frequency-weighted IoU, MCC, Cohen's kappa |
| Probability diagnostics | Sampled one-vs-rest ROC AUC and AP, multiclass Brier, NLL, 10-bin ECE |
| Continual validation | Stage/domain mIoU matrix, average seen-domain score, forgetting and backward transfer |

Background is included when it is a configured class. A class absent from both prediction and truth is excluded from the mean; false positives for an absent ground-truth class yield IoU zero and remain included. Ignore/padded pixels are excluded. Boundary IoU excludes a band adjacent to void/padding, preventing artificial boundaries there; it is not a Boundary F1 metric. Metrics use the resized/padded model grid's valid pixels, not a native-resolution geospatial area weighting.

Testing writes:

```text
outputs/<run>/test/final/
  metrics.json, classwise_metrics.csv, per_image_metrics.json
  domain_metrics.json, provenance.json
  confusion_counts.png, confusion_normalized.png, classwise_scores.png
  prediction_000.png ...    # image / truth / prediction / error / confidence
  roc_sampled.png, precision_recall_sampled.png, calibration_sampled.png
  sampled_probability_metrics.json
  tsne_class.png, tsne_domain.png, tsne_points.npz, tsne_metadata.json
```

Full confusion-derived metrics and Boundary IoU use all valid test pixels. Probability curves use a reproducible bounded uniform pixel reservoir. t-SNE uses a separate bounded uniform reservoir of adapted SAM spatial features, PCA reduction, then seeded t-SNE; labels are nearest-resized ground truth at feature locations. Sample counts/settings are saved. Rare classes may be absent from a uniform sample. These plots are exploratory and do not prove feature separability or quantum benefit. Fewer than four tokens produces an explicit skipped status.

Training also saves `history.csv`, `training_curves.png` and `continual_matrix.png`. Test reporting includes `per_image_distributions.png` and `classwise_iou_distributions.png`. Testing is explicit and never runs automatically each epoch. Existing test reports are protected; use `--output` for another report.

## A100 40 GB settings

Start with current batch **2**, replay batch **2**, evaluation batch **2**, **4 data workers**, pinned memory, persistent workers, and prefetch factor 2. CPU core count and storage throughput, not VRAM, determine worker suitability. Increase workers to eight only after profiling the actual host. Gradient accumulation is configurable and defaults to one.

SAM/decoder use CUDA BF16 autocast; the circuit disables autocast and uses its parameter precision. TF32 is enabled. Frozen SAM runs without encoder gradients. The primary PennyLane backend is `default.qubit` with PyTorch backprop; the code does not mislabel it as `lightning.gpu`. Spatial circuit inputs are chunked (32 initially), with activation checkpointing to bound retained gate graphs. Backward recomputation trades time for memory. Actual circuit execution placement and speed need A100 profiling.

Default disk caching stores frozen SAM features in fp16 keyed by preprocessed image content and checkpoint hash. It avoids repeating expensive encoder passes after the first encounter. PCA reads only first-domain training features even if other cache files exist. Cache files are reusable storage, **not** unrestricted historical training samples; replay access remains byte capped. Cache IO/hash cost and CPU metric computation can become bottlenecks.

Online image flips require `dataset.augment: true` and `model.cache_dir: null`. Flipping cached features would not equal re-encoding a flipped image and is deliberately rejected. Disk caching therefore makes the default augmentation policy explicit: no random image flips.

`stage0.py` profiles synthetic feature updates without claiming an end-to-end training speed. Full epoch times include data, metrics and validation separately in history. Do not use the microbenchmark alone as the 48-hour research cost forecast. No automatic OOM fallback changes batch size or the experimental protocol.

## Classical controls

Supported `model.adapter`: `quantum`, `frozen_quantum`, `no_entanglement`, `sine`, `mlp`, `classical`, `none`. `classical` is unconstrained 256→r→256; `mlp` shares PCA→8 with the circuit. Change `rank` for width ablations. Controls are independent configurable runs, not automatically selected winners.

```bash
python train.py --config configs/openearthmap_classical.yaml
python train.py --config configs/openearthmap_no_adapter.yaml
```

Use matched split/seed/domain order/update budgets when comparing models. Rank-regularized and continual-backprop baselines, the full supervised anchor, six-arm plasticity experiments, and confirmatory statistical selection are not yet implemented; the complete research roster cannot be claimed.

## Provenance

The data/config/reporting organization was informed by the user's [reference repo](https://github.com/Arun2005-srm/skin_lesion_segmentation_XAI), inspected at `418fd4bfd567c7678e5488d05e8ea25e4b6395e2`. This implementation is newly written and does not import its architecture.

OEM labels were checked against [the authors' class mapping](https://github.com/bao18/open_earth_map/blob/ad645629af37369dfbe481be40515a587c7cd4af/open_earth_map/utils.py). LandCover.ai labels follow [the dataset paper](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Boguszewski_LandCover.ai_Dataset_for_Automatic_Mapping_of_Buildings_Woodlands_Water_and_CVPRW_2021_paper.pdf). The local mask audit remains mandatory for each downloaded release.
