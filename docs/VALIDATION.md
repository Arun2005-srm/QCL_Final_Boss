# Local validation

Date: 2026-10-01. Execution host: Windows, CPU only. Installed packages are recorded in `tested-cpu-environment.txt`.

## Automated checks

`python -m pytest -q` passes the data/model/reporting tests. Coverage includes:

- Reproducible group-isolated 70/15/15 partitioning.
- Raw-label rejection and dataset-specific mapping replacement.
- Correct ignored-pixel, absent-class and false-positive behavior.
- Boundary IoU identity and boundary-shift response.
- Segmentation loss with absent classes and all-ignored targets.
- Chunked versus unchunked PennyLane outputs and parameter gradients in FP64.
- Replay byte limits, unique arrival handling and restored sampling state.
- End-to-end synthetic training, checkpoint resume and held-out report generation.
- Two-domain training, replay use and stage checkpoints.
- Cached-feature content keys and augmentation/cache incompatibility.
- Identical decoder initialization across paired adapter variants.

The generated t-SNE figure was visually inspected. Test reports also contain confusion plots, prediction panels, class distributions, ROC/PR curves and a reliability diagram. Synthetic plots are diagnostic artifacts, not remote-sensing results.

## Real architecture check

Instantiating the installed original SAM ViT-B implementation with random weights confirmed:

- Image encoder parameters: 89,670,912.
- Input shape: `[1,3,1024,1024]`.
- Encoder output shape: `[1,256,64,64]`.
- Primary adapter parameters: 2,080 (32 circuit angles + 2,048 projection weights).

This was a structural forward check only. It did not use the pretrained checkpoint and says nothing about segmentation accuracy.

## Circuit numerical evidence

`circuit-numerics-cpu.json` stores the actual seed, 64 normalized angle inputs, circuit parameters, singular values and gradient result. The stacked FP64 Jacobian was `[512,32]` and had rank 32 at both relative thresholds `1e-8` and `1e-6`. Parameter gradients were finite and nonzero. The measured gradient norm for the declared squared-output test was about 14.67.

This establishes local numerical sensitivity for that test point. It does not prove global identifiability, sustained plasticity, quantum advantage, or useful branch influence during training.

## Still to measure

Pretrained-SAM real-data behavior, raw downloaded dataset feasibility, A100 execution placement/peak memory/throughput, full-epoch resource costs, and all scientific accuracy/retention outcomes remain unmeasured. No weights or datasets are bundled. The implementation status document lists the broader research experiments that are not provided by this training pipeline.
