# Implementation status and protocol amendment

Date: 2026-10-01. This file distinguishes executable software from the broader reviewed research plan.

## User-requested changes

- Train/validation/test is now 70/15/15 by independent scene groups within each domain. This overrides the archived v1.1 four-part split.
- Epoch-level console, JSON and CSV reporting includes train/validation loss and accuracy, Dice, micro IoU, mIoU, Boundary IoU and class IoUs.
- Held-out reports include full confusion-derived metrics, boundary metrics, per-image/domain reports, probability plots and t-SNE.
- OpenEarthMap runs first; LandCover.ai is a separate configured experiment because their class vocabularies differ.
- Initial A100 40 GB settings: current batch 2, replay batch 2, eval batch 2, four workers, BF16 classical layers, bounded quantum chunks, reusable frozen-feature cache.

## Implemented and exercised

- New configurable data pipeline: CSV or paired folders, explicit integer/color masks, label audit, scene-group isolation, saved content hashes, source-preserving tiling.
- SAM ViT-B encoder wrapper and reviewed eight-qubit RX→RY/CNOT/X-readout adapter, frozen first-domain PCA, residual calibration, spectral output constraint, three-convolution decoder.
- Real PennyLane analytic circuit gradients, chunking and activation-checkpoint recomputation.
- No-adapter, PCA-MLP, bounded sine, frozen circuit, non-entangling circuit, and unconstrained classical adapter variants.
- Adam training with current-domain segmentation plus byte-capped DER++-style replay, unique-arrival reservoir, immutable insertion logits, and no boundary optimizer reset.
- Stage/domain training, fixed update cap, all-seen validation, endpoint checkpoints, epoch-boundary resume, validation forgetting and backward transfer.
- Held-out test command with frozen split/checkpoint provenance; scientific test reports never select training hyperparameters automatically.
- CPU tests for data isolation, label failures, metrics, boundary behavior, loss gradients, quantum gradient equivalence, replay accounting, end-to-end training/report generation/resume, and two-domain operation.

## Measurement limits

The real installed SAM architecture was instantiated with random weights **only to check structure**: 89,670,912 image-encoder parameters and output `[1,256,64,64]` for `[1,3,1024,1024]`. The eight-qubit adapter has 2,080 parameters including the output projection. The eight-class decoder adds 369,544; trainable total is 371,624. No pretrained SAM accuracy was measured.

End-to-end tests use the explicitly named synthetic encoder and tiny generated images. They verify software execution, not remote-sensing competence. No OpenEarthMap/LandCover.ai data, trained checkpoint, A100 measurements, or accuracy claim is included in the repository.

`stage0.py` supplies circuit numerical checks and adapter/decoder microbenchmarking. `audit_dataset.py` supplies the real-data audit. The full Stage 0 evidence package remains pending actual data/checkpoint paths and execution on the target host.

## Research components not supplied by this training pipeline

The following remain requirements of the archived research plan and are **not implemented as completed scientific experiments** here:

- A source-verified L2-ER port and faithful continual-backprop control.
- The complete classical candidate-search roster and strongest-comparator selection.
- The supervised U-Net competence-anchor experiment.
- The six reset/plasticity probes with independently calibrated targets, censoring analysis and diagnostic partitions.
- Paired multi-seed confirmatory inference, power planning, Holm testing and automatic research gate decisions.
- Finite-shot/noisy training and the corresponding memory/gradient validation.

The revised 70/15/15 workflow does not itself resolve independent development/calibration/diagnostic isolation. It is suitable for engineering development and descriptive continual-learning evaluation; it must not be presented as passing the archived confirmatory protocol. No component silently substitutes for an unavailable research control.

## Interpretation and reproducibility

The runtime configuration and installed versions are saved per run. `tested-cpu-environment.txt` records the local test environment; it is evidence, not a CUDA installation recipe. Pin a compatible CUDA environment and verify on the A100 before resource forecasts. Do not advertise measured speed or superior metrics from synthetic tests.

The default feature cache disables random image augmentation. Online augmentation is supported only with caching off. This choice is explicit because applying a flip to a cached SAM feature is not equivalent to encoding a flipped image.

Approximate group ratios, source-overlap checks, ignored labels, conditional undefined metrics, sampled probability curves and t-SNE caveats are documented in the README. The word universal refers to the configurable RGB semantic-segmentation data contract, not arbitrary sensor/modal/task support.
