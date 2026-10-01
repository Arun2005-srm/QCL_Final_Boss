# SAM-Based Quantum Continual Segmentation — Implementation Specification

**Version:** 1.1  
**Date:** 1 October 2026  
**Status:** Proposed implementation specification for feasibility review. Training and feasibility results are not yet established.  
**Hardware target:** One NVIDIA A100 with 40 GB VRAM.  
**Backbone:** Original SAM ViT-B only.

This document consolidates the latest implementation decisions. It supersedes earlier chat drafts where they differ. Dataset-specific domain assignments, measured runtime, and confirmatory sample size remain outputs of the feasibility phase; they are not invented here.

### Version 1.1 pre-run amendments

This revision incorporates the final-review requirements into the operative sections, not just an appendix. It pins the residual-cap calculation and failure rule (§8), finer 2,000-update plasticity probes (§19), the primary comparison (§21), the split fallback (§12), a supervised competence anchor (§22.1), the noise convention/device/gradients (§24), and staged cost limits (§28).

Stage 0 consists only of the data audit, numerical checks, tiny-subset overfit, and circuit profiling. It cannot establish a research gate pass. Gate-bearing work requires a saved pre-run manifest naming the selected split branch, comparator-search roster, resource forecast, and all evaluation rules below. No experiments are claimed to have run merely because this specification is updated.

## 1. Objective and experimental boundaries

**Research question:** Does a trainable quantum-circuit residual adapter improve sustained adaptation of frozen SAM features while retaining earlier segmentation performance, compared with strong classical alternatives?

The task is domain-incremental semantic segmentation:

- Geographic domains arrive sequentially.
- Every domain uses the same eight land-cover classes.
- One shared adapter and decoder learn throughout the sequence.
- Model capacity remains fixed as domains arrive.
- No domain identifier is supplied during inference.
- Historical training access is restricted to the replay allowance.
- Historical test data is available only to the evaluator.

The potential claim is an **empirical architectural benefit over the tested alternatives**. An eight-qubit simulation does not establish quantum computational advantage.

Excluded from this version:

- DINO or other replacement backbones.
- SAM's prompt encoder and native prompt-conditioned mask decoder.
- Per-domain expert banks, routing, or growing model capacity.
- LoRA inside SAM.
- EWC as the proposed mechanism.
- Amplitude encoding.
- Training the SAM encoder.
- Quantum hardware speedup claims.

## 2. Complete model architecture

```text
RGB image/tile                            [B, 3, 1024, 1024]
  |
Frozen SAM ViT-B image encoder
  |
F                                        [B, 256, 64, 64]
  |---------------------------------------------------|
  |                                                   |
Adaptive average pooling                 [B, 256, 16, 16]
  |
Frozen centered PCA, 256 -> 8             [B, 16, 16, 8]
  |
Vector normalization and angle scaling   [B*256, 8]
  |
Shared eight-qubit circuit
Two RX-then-RY/CNOT blocks; 32 angles
  |
Eight Pauli-X expectations               [B*256, 8]
  |
Trainable bias-free Linear(8, 256)        [B, 256, 16, 16]
  |
Bilinear upsample + fixed residual scale [B, 256, 64, 64]
  |                                                   |
  |---------------------- ADD <-----------------------|
                           |
F'                                       [B, 256, 64, 64]
                           |
Custom semantic decoder: 256 -> 128 -> 64 -> 8
                           |
Coarse logits                            [B, 8, 64, 64]
                           |
Bilinear upsample                        [B, 8, 1024, 1024]
                           |
Argmax over classes                      [B, 1024, 1024]
```

The circuit transforms pooled feature channels. It does not encode the entire image into eight qubits. Spatial locations share circuit parameters but are evaluated independently; the circuit does not entangle different spatial locations.

## 3. SAM image encoder

| Property | Specification |
|---|---|
| Encoder | Original SAM ViT-B image encoder |
| Registry identifier | `vit_b` |
| Checkpoint | `sam_vit_b_01ec64.pth` |
| Input | RGB, `[B,3,1024,1024]` |
| Patch embedding | Conv2d, kernel 16, stride 16 |
| Transformer token grid | 64 x 64 = 4,096 tokens |
| Transformer width | 768 |
| Transformer blocks | 12 |
| Attention heads | 12 per block |
| Head dimension | 64 |
| MLP width | 3,072 |
| MLP activation | GELU |
| Attention pattern | Eight window-attention blocks, four global-attention blocks |
| Global-attention blocks | 3, 6, 9, 12, using one-based numbering |
| Window size | 14 x 14 tokens |
| Neck | 1x1 convolution, LayerNorm2d, 3x3 convolution, LayerNorm2d |
| Neck channels | 768 -> 256 -> 256 |
| Encoder output | `[B,256,64,64]` |
| Encoder parameters | 89,670,912, all frozen |

Each transformer block contains two residual sublayers: normalized attention and a normalized two-linear-layer MLP. Load the checkpoint strictly, record its checksum, retain the image encoder, and discard the prompt encoder and native mask decoder. Assert the parameter count in the implementation.

Sources: [SAM builder](https://raw.githubusercontent.com/facebookresearch/segment-anything/main/segment_anything/build_sam.py), [encoder implementation](https://raw.githubusercontent.com/facebookresearch/segment-anything/main/segment_anything/modeling/image_encoder.py).

## 4. Input preprocessing and geometry

Primary model input: a 1024 x 1024 RGB tile.

Apply SAM normalization exactly once to RGB values on the 0-255 scale:

```text
mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]
x_normalized[c] = (x[c] - mean[c]) / std[c]
```

- Preserve existing 1024 x 1024 tiles.
- Extract 1024 x 1024 windows from larger scenes.
- Normalize before zero-padding smaller edge windows.
- Assign padded and unlabeled mask pixels to ignore ID 255.
- Retain valid height and width for evaluation.
- Never bilinearly resize categorical masks.
- Reject image-mask alignment and geospatial-grid mismatches.
- Require explicit RGB band selection for multispectral sources.
- Do not divide inputs by 255 and then apply the constants above.

Primary evaluation uses non-overlapping windows. A future overlapping-window mode must average probabilities before source-scene scoring.

Source: [SAM preprocessing](https://raw.githubusercontent.com/facebookresearch/segment-anything/main/segment_anything/modeling/sam.py).

## 5. Residual adapter definition

For frozen encoder features F:

```text
F_adapted = F + alpha * Upsample_64(W_out * Q_theta(A(F)))
```

Here A comprises spatial pooling, frozen PCA, and angle normalization; Q is the shared circuit; W_out is a trainable 256 x 8 matrix; and alpha is a scalar frozen after initialization. Use bilinear upsampling with `align_corners=False`.

| Spatial grid | Circuit instances/image | Use |
|---|---:|---|
| 4 x 4 | 16 | Execution smoke test and coarse-resolution ablation |
| 16 x 16 | 256 | Main feasibility candidate |
| 64 x 64 | 4,096 | Profile first; train only if affordable |

The untouched feature path preserves the original 256-channel, 64 x 64 SAM representation. Only the residual correction is pooled.

## 6. Frozen PCA and angle formation

At each pooled location, f has 256 components. Store the first-domain training mean mu and eight orthonormal PCA directions P:

```text
mu: [256]
P:  [256, 8]
z = transpose(P) @ (f - mu)
angles = (pi / 2) * z / max(norm(z, 2), 1e-6)
```

Every angle lies in [-pi/2, pi/2]. The zero vector maps to zero angles.

PCA fitting:

- Fit exclusively on first-domain training features.
- Sample at most 100,000 valid pooled locations deterministically.
- Accumulate statistics in FP64.
- Do not whiten.
- Freeze the mean and components for the full sequence.
- Save sampling seed, fitted tensors, and explained variance.
- Make each component's largest-magnitude loading positive to fix sign ambiguity.
- Share the exact PCA tensors across paired mechanism comparisons.

Each domain order fits PCA on its own first domain. No future-domain features contribute. The normalization deliberately discards magnitude; record this limitation.

Primary quantum and matched mechanism models use frozen PCA. Practical classical competitors are not forced through PCA. Random orthogonal projection and trainable projection are separate ablations. A tanh-versus-normalized-encoding comparison must use the same trainable projection setup in both arms.

## 7. Quantum circuit

Initialize eight qubits in the all-zero state. Apply RY angle encoding, followed by two trainable blocks. Each block applies RX then RY to every qubit, then CNOTs 0->1, 1->2, ..., 6->7, in that order. Measure eight Pauli-X expectations.

```python
def circuit(angles, theta):
    # angles: [..., 8]
    # theta: [2, 8, 2]
    for q in range(8):
        qml.RY(angles[..., q], wires=q)

    for layer in range(2):
        for q in range(8):
            qml.RX(theta[layer, q, 0], wires=q)
            qml.RY(theta[layer, q, 1], wires=q)
        for q in range(7):
            qml.CNOT(wires=[q, q + 1])

    return tuple(qml.expval(qml.PauliX(q)) for q in range(8))
```

| Quantity | Value |
|---|---:|
| Qubits | 8 |
| Encoding rotations | 8 |
| Trainable blocks | 2 |
| Trainable rotations/block | 16 |
| Trainable circuit angles | 32 |
| CNOTs/block | 7 |
| Total CNOTs | 14 |
| Logical gates before measurement | 54 |
| Output observables | 8 |
| Statevector amplitudes | 256 complex values |

All X observables share one measurement basis. Hardware-style execution additionally requires basis changes; these are not included in the gate count above. This is not a transpiled physical-gate count.

Initialize angles uniformly in [-0.1, 0.1], not exactly at zero.

**Gate-order correction:** RX then RY supersedes earlier RY-then-RX drafts. A small numerical Jacobian check found eight unobservable parameter directions with the old ordering and none at the tested input with the new ordering. This does not establish global identifiability or trainability. Repeat checks across random inputs in implementation tests.

**Rank convention:** For one input the output Jacobian is 8 x 32 and has rank at most eight. A reported rank of 32/32 must refer to a Jacobian stacked over multiple inputs; nonzero columns alone do not establish full rank. In Stage 0, stack 64 seeded normalized input vectors, use FP64, report all singular values, and report ranks at relative thresholds 1e-8 and 1e-6 times the largest singular value. Record inputs and initialization. Do not present a reviewer's rank report as an independently reproduced result. The non-entangling control can have structural redundancy: with one RY input encoding and one final scalar readout per qubit its input dependence has only sine/cosine coefficients, so its effective function dimension need not equal its 32 stored angles. Report this limitation; the other classical controls remain necessary.

For an isolated RY-encoded qubit, X expectation is sin(angle), preserving signed information on the chosen interval. Test the full circuit separately because entanglers change this relationship.

## 8. Output projection and residual calibration

Use `Linear(8,256,bias=False)`: 2,048 trainable parameters.

Initialize its eight columns orthonormally. After each optimizer update, enforce spectral norm at most one:

```text
W_out = W_out / max(1, largest_singular_value(W_out))
```

Calibrate alpha once on up to 32 first-domain training tiles:

```text
alpha = 0.01 * RMS(F) / max(RMS(Upsample(W_out * Q_theta)), 1e-6)
```

Compute RMS over valid locations. Alpha is nonzero and is not trainable. The initial residual is approximately 1% of feature RMS; this is an initialization rule, not a requirement that the trained residual remain at 1%.

Log residual/feature RMS, output-projection spectral norm, circuit and projection gradient norms, and circuit-output variance. The decoder remains unconstrained by these branch-specific bounds.

### Residual cap and predeclared failure rule

Let s = RMS(F) on the calibration tiles and r0 = RMS(Upsample(W_out * Q_theta)) before multiplying by alpha at initialization. Since each of eight expectations lies in [-1,1], norm(Q_theta) <= sqrt(8). With spectral_norm(W_out) <= 1, the 256-channel RMS of the output is at most sqrt(8)/16. Bilinear interpolation is a convex combination, so this per-location norm bound also holds after interpolation.

```text
absolute_residual_RMS_bound = abs(alpha) * sqrt(8) / 16
relative_bound_on_calibration = absolute_residual_RMS_bound / s
                              = 0.01 * sqrt(8) / (16 * r0)
```

The last expression assumes r0 exceeds the calibration epsilon; otherwise use the original alpha formula directly. A roughly 3% cap is plausible for some initializations, but is not a universal numerical bound. Calculate the actual bound for each seed before training. On other domains divide the same absolute bound by that domain's measured feature RMS. The branch is intentionally capacity-limited, and its allowed correction may be only a few percent of feature RMS.

Log the realized ratio at initialization, every 50 training updates, and each domain endpoint, including distribution across examples. Record the calculated upper bound beside it. Require finite, numerically verifiable circuit gradients in Stage 0; a small nonzero gradient is not evidence of a useful learning effect.

If a correct implementation fails the >=1-point branch-dependence gate, stop this candidate at the declared scale. Do not increase alpha, relax the projection bound, or change width to rescue that gate. A verified implementation bug permits a logged correction and restart. A different scale is a new protocol version and separate experiment, not a continuation of the failed candidate.

## 9. Semantic decoder

The decoder operates entirely on the 64 x 64 feature grid until final interpolation.

| Order | Operation | Output | Parameters |
|---|---|---|---:|
| 1 | Conv2d 256->128, 3x3, padding 1, no bias | `[B,128,64,64]` | 294,912 |
| 2 | GroupNorm, 16 groups, affine | Same | 256 |
| 3 | GELU | Same | 0 |
| 4 | Conv2d 128->64, 3x3, padding 1, no bias | `[B,64,64,64]` | 73,728 |
| 5 | GroupNorm, 8 groups, affine | Same | 128 |
| 6 | GELU | Same | 0 |
| 7 | Conv2d 64->8, 1x1, bias | `[B,8,64,64]` | 520 |
| 8 | Bilinear interpolation | `[B,8,1024,1024]` | 0 |

Decoder parameters: **369,544**.

GroupNorm epsilon: 1e-5. No BatchNorm, dropout, or intermediate-SAM skip connections in the primary model. Use Kaiming convolutional initialization, normalization scale one and offset zero, and zero classifier bias.

Full-resolution output is interpolated from coarse logits. Upsampling cannot reconstruct information the representation failed to preserve.

## 10. Parameter and layer accounting

| Component | Frozen parameters | Trainable parameters |
|---|---:|---:|
| SAM image encoder | 89,670,912 | 0 |
| PCA mean/basis | Buffers | 0 |
| Circuit | 0 | 32 |
| Output projection | 0 | 2,048 |
| Decoder | 0 | 369,544 |
| Residual scalar | Buffer | 0 |
| **Total** | **89,670,912** | **371,624** |

Total parameters excluding buffers: **90,042,536**. Circuit angles constitute approximately **0.0086% of trainable parameters**. Attribution tests are mandatory.

FP32 weight storage: approximately 342.07 MiB for SAM alone, or 343.49 MiB including adapter and decoder. Optimizer state, activations, buffers, replay, and library overhead are additional.

Explicit layer counts:

- One patch-embedding convolution.
- Twelve transformer blocks.
- Two encoder-neck convolutions.
- Two quantum blocks.
- One adapter output projection.
- Three decoder convolutions.

Do not combine these different operation types into a misleading single depth value.

## 11. Dataset and labels

Primary dataset: OpenEarthMap.

| Model ID | Class |
|---:|---|
| 0 | Bareland |
| 1 | Rangeland |
| 2 | Developed space |
| 3 | Road |
| 4 | Tree |
| 5 | Water |
| 6 | Agriculture land |
| 7 | Building |

Verify raw mask IDs against the downloaded release before mapping. Do not assume raw zero means model class zero. Internal stored masks use uint8 labels 0-7 and ignore value 255; convert to integer class indices for the loss.

Required manifest fields:

```text
sample_id,image_path,mask_path,source_scene_id,spatial_group_id,
domain_id,split,valid_height,valid_width
```

Retain georeferencing and window coordinates where available.

Source: [OpenEarthMap vocabulary and data description](https://open-earth-map.org/overview_oem.html).

## 12. Domains and splits

Construct geographic domains using dataset metadata. Audit labeled-data availability, independent scenes, overlaps, duplicates, class coverage, valid-pixel counts, and partition sizes.

Target ten qualifying domains. Do not create nominal domains by arbitrarily fragmenting a small number of scenes.

Proposed within-domain group split:

- Training: 60%.
- Calibration: 20%.
- Diagnostic validation: 10%.
- Test: 10%.

Require at least two independent groups in each evaluation partition. Indivisible groups may prevent exact percentages. Calibration supports fresh-reference thresholds and development selection; diagnostic validation supports plasticity probes. Test outcomes must not select hyperparameters, stopping decisions, or model variants.

Reserve separate development domains where the data supports them. Publish final domain identities and manifests after the audit. No future-domain images fit PCA, normalization, or other current-stage statistics.

### Deterministic split fallback

Select the split branch from counts/geometry only, before any model performance is observed. A group is an independent source scene or a connected set of spatially overlapping source scenes; tiles and augmentation views never count as independent groups. Regions sharing a group must remain on the same development/confirmation side or be excluded. Do not split a group to satisfy a minimum.

1. **Preferred four-way split:** For regions with at least 20 groups, allocate test = max(2, floor(0.10*N)), diagnostic = max(2, floor(0.10*N)), calibration = max(2, floor(0.20*N)), and training = the remaining groups. Assign using a saved seed and sorted group IDs. Require at least eight training groups.
2. **Fallback A — sparse regions:** For 12-19 groups, reserve two test groups and use five prespecified three-way internal assignments on the remaining groups. Each assignment reserves two calibration and two diagnostic groups, with the rest for training (at least six). Within an assignment the partitions are disjoint. Use this regime for all selected domains in that stream; do not mix it silently with the preferred regime. The repetitions are internal folds, not five extra independent statistical runs. Average fold-level metrics within a run. Refit PCA only on that fold's permitted first-domain training partition. Test groups remain untouched. Apply this fallback only if its multiplied cost fits the forecast.
3. **Fallback B — fewer domains:** Exclude regions with fewer than 12 independent groups. If ten domains are unavailable, use all qualifying confirmation domains down to a minimum of five and report the reduced scope. Do not relabel subdivisions as new geographic domains.
4. **Stop:** If fewer than five confirmation domains or three separate development domains remain, or Fallback A is unaffordable, report the data protocol as infeasible. Stop gate-bearing experiments. Changing datasets or weakening isolation requires a new specification.

Use qualifying-region count descending and region ID ascending to break ties deterministically. Reserve three qualifying regions for development and up to ten remaining regions for confirmation; record the resulting selection. The sample-size minimum is a feasibility rule, not proof that two evaluation groups give a precise regional estimate. Publish group counts and uncertainty. Verify actual raw mask IDs against the downloaded files; an unverified label map blocks even Stage 0 training.

The distinct-domain stream is primary. A recurring-domain stress test is separate and must report repeated exposures and time since last exposure. Do not label 100 repeated arrivals as 100 independent domains.

## 13. Feature caching and augmentations

Run SAM in evaluation mode without encoder gradients. Cache FP16 features `[256,64,64]`, uint8 masks `[1024,1024]`, and geometry metadata. Cast features to FP32 for adapter/decoder training.

Each feature map is 2 MiB.

Use predetermined paired geometric views, beginning with identity and horizontal flip for development comparisons. Encode every image view separately through SAM. Flipping cached features is not assumed equivalent to encoding the flipped image.

All methods share view schedules. Replay retains one selected view per source sample; extra views consume extra memory.

Historical caches may remain on disk for reproducibility, but the training loader must reject access outside the permitted buffer. Evaluation caches are never training inputs.

## 14. Replay records and selection

Primary budget: **128 MiB**. Secondary: **32 MiB and zero replay**.

| Retained item | Bytes |
|---|---:|
| FP16 feature map | 2,097,152 |
| uint8 mask | 1,048,576 |
| FP16 logits `[8,64,64]` | 65,536 |
| Fixed metadata allowance | 512 |
| **Record total** | **3,211,776** |

Capacities: 41 records at 128 MiB; 10 records at 32 MiB. Reject oversized metadata. Count uncompressed tensor bytes plus metadata; compression does not increase capacity.

Ordinary replay reserves the same logit field even when unused, preserving identical sample capacity across loss comparisons.

- Reservoir-sample unique arriving source samples.
- Register a sample once, not once per epoch.
- Pair selection decisions across model families.
- Store detached logits upon first buffer entry.
- Do not update stored targets using future model states.
- Store no autograd graphs.
- Earlier current-domain examples may enter memory; future-domain examples may not.

## 15. Loss functions

Current supervised loss:

```text
L_seg = L_cross_entropy + 0.5 * L_soft_dice
```

Cross-entropy uses upsampled logits, ignore ID 255, and initially uniform class weights. Keep all eight output classes in the softmax. Missing classes do not justify changing the label space. An all-ignored sample contributes no supervised loss.

Dice uses softmax probabilities, masks invalid pixels, uses epsilon 1e-6, and averages over ground-truth-present classes in the minibatch. Cross-entropy still penalizes predictions of absent classes.

Ordinary replay:

```text
L = L_seg_current + L_seg_replay
```

DER++-style segmentation adaptation:

```text
L = L_seg_current + beta * L_seg_replay + lambda_logit * L_logit
beta = 1.0
lambda_logit = 0.1  # initial development value
```

L_logit is mean-squared error between current and stored 64 x 64 logits, weighted by the valid-pixel fraction at each coarse location. Report this as a segmentation adaptation, not an unchanged reproduction of the original classification method.

## 16. Optimization and update budgets

| Setting | Initial development value |
|---|---|
| Classical optimizer | Adam |
| Circuit optimizer | Adam, separate parameter group/state |
| Decoder/projection learning rate | 1e-3 |
| Circuit learning rate | 1e-2 |
| Betas | (0.9, 0.999) |
| Epsilon | 1e-8 |
| Default weight decay | 0; regularized arms specify separately |
| Current minibatch | 2 |
| Replay minibatch | 2 when available |
| Gradient accumulation | 1 initially |
| Classical training precision | FP32 |
| LR schedule | Constant within and across domains |
| Boundary optimizer reset | None |
| Boundary model reset | None |
| Default gradient clipping | Disabled |

Fail explicitly on nonfinite loss/gradients. Reset diagnostics use separate model copies.

Update budgets:

- Tiny-subset overfit: up to 500 updates.
- Execution/engineering screening: 200 updates/domain; not used to select a scientific winner or assess plasticity/retention gates.
- First-domain competence: 2,000 updates.
- Initial full-stream design: 2,000 updates/domain.

Use fixed-update endpoints for primary comparisons. Do not give one method best-historical-checkpoint selection while evaluating another at its final update.

The candidate search ceiling is 16 configurations per family, not a commitment to execute that matrix. Scientific hyperparameter ranking and final development selection use the same 2,000 updates/domain as confirmation. A 200-update run may expose crashes or estimate cost, but cannot choose the winner. Before comparing scores, use the §28 cost forecast to lock the same affordable number of configurations for every competing family. Preserve sample-exposure and timing records. If the required roster and five paired development runs do not fit, the comparison remains incomplete; do not call the best affordable subset the strongest competitor across both tiers.

## 17. Comparison models

### 17.1 Mechanism controls

Share PCA inputs, grid, decoder, projection dimensions, and residual calibration.

| Control | Definition |
|---|---|
| No adapter | Decoder directly consumes SAM features |
| Classical MLP | Eight inputs -> hidden layer -> eight outputs |
| Bounded sinusoidal | Two elementwise sine stages |
| Orthogonal | Trainable orthogonal mixing plus bounded nonlinearity |
| Frozen circuit | Same initial circuit as paired run, all 32 angles frozen |
| Non-entangling circuit | Same 32 rotations and readout, only CNOTs removed |
| Trainable circuit | Full specified circuit |

An exact 32-parameter sinusoidal comparator is:

```text
h = sin(s1 * a + b1)
q = sin(s2 * h + b2)
```

Each parameter vector has eight entries. Its coefficient bounds and projection rule must be recorded in the resolved configuration. Frozen-circuit models retain fewer trainable parameters by design; report this instead of adding dummy parameters.

### 17.2 Practical competitors

- Standard residual adapter: 256 -> r -> 256.
- Candidate widths: 4, 16, 32, 64.
- Pooled and full-grid execution where affordable.
- Plasticity-maintaining classical variants.
- Larger semantic decoder without an external adapter.

With input bias and bias-free output, the standard adapter has 513*r parameters. Width four has 2,052 parameters, close to the quantum adapter's 2,080.

The larger-decoder control inserts a bias-free residual 128 -> 8 -> 128 channel block inside the decoder, adding 2,048 parameters.

Select the strongest classical competitor across both tiers on development data. Winning only against eight-dimensional restricted controls is insufficient for continuation.

## 18. Classical plasticity regularizers

### 18.1 L2 plus activation effective rank

For the segmentation adaptation, sample at most 1,024 valid spatial activation vectors/update from the adapter hidden representation and both decoder hidden representations. For each activation matrix H:

```text
sigma = singular_values(H)
p = sigma / max(sum(sigma), 1e-8)
effective_rank = exp(-sum(p * log(p + 1e-8)))
```

Average negative effective rank across selected layers. Apply one additional rank-gradient update per supervised update. Apply L2 to classical weight tensors, excluding biases and normalization affine parameters. Search L2 strength and rank-step size separately; count extra computation.

Record activation sampling, selected layers, coefficients, and update ordering. The author implementation uses activation effective rank and separate rank updates; weight-spectral regularization is a different method. Label this a segmentation adaptation.

Reference: [author implementation](https://github.com/KevinGuo27/lop-jax/blob/main/permuted_mnist/train_permuted_mnist.py).

Source-verification note: the source inspected during specification preparation used singular values of the activation matrix in `effective_rank`, then separate rank-gradient updates. Eigenvalues of H-transpose-H are squared singular values and produce a different normalized spectrum; do not substitute them silently. Fetch and pin a source commit before implementation, and compare losses and gradients on fixed small matrices. Until that check passes, mark the port unverified. A failing or unavailable source check blocks claims of faithful implementation, not the unrelated Stage 0 circuit checks.

### 18.2 Continual-backpropagation adaptation

For the classical residual MLP, track hidden-channel age and utility based on centered activation magnitude, outgoing-weight magnitude, and incoming-weight magnitude.

- EMA decay: 0.99.
- Initial maturity threshold: 100 updates.
- Candidate replacement rates: 1e-4, 1e-3, 1e-2.
- Replace least-useful eligible units.
- Reinitialize incoming weights and zero outgoing weights.
- Clear affected optimizer moments.
- Log every replacement.

An extension to convolutional decoder channels is not automatically equivalent to the original neuron method. The exact utility formula, replacement accounting, and optimizer handling must be pinned to a reference and tested before labeling the baseline a reproduction. This remains an implementation validation requirement, not an established equivalence.

## 19. Plasticity and attribution probes

At selected stages create independent copies:

1. Unchanged sequential learner.
2. Fresh adapter and decoder.
3. Adapter reset only.
4. Decoder reset only.
5. Optimizer reset only.
6. Joint adapter/decoder reset.

All copies receive identical current-domain minibatches and update budgets. Encoder, PCA, preprocessing, and fixed residual scale remain unchanged. Adapter reset reinitializes circuit and output projection. Reset optimizer state only for reset modules, except the optimizer-only intervention. Diagnostic copies never replace the main learner.

Define the domain target using five independent fresh-reference calibration runs, each trained for 2,000 updates. Use five saved seeds disjoint from diagnostic and confirmatory seeds. For each paired diagnostic, the fresh reference has the same architecture and input projection as the sequential learner. All calibration and diagnostic training uses only that domain's training partition; their evaluation partitions are disjoint under §12. Freeze the selected training hyperparameters before these reference runs. Define:

```text
target[d] = 0.9 * mean(final_calibration_mIoU_of_fresh_references[d])
```

Save the target, calibration-run IDs and final scores before diagnostic evaluation. Each fresh/reset/sequential probe receives 2,000 updates. Evaluate at update zero and every five updates through update 2,000. This fine evaluation cadence applies to probes; ordinary training logs may remain less frequent. Include its substantial evaluation cost in the budget.

For a crossing require two consecutive scores >= target. If the first of these is at t1 and its preceding evaluation t0 is below target, estimate crossing time by linear interpolation between (t0,score0) and (t1,score1):

```text
t_cross = t0 + (target-score0) * (t1-t0) / (score1-score0)
log_ratio = log(t_cross_sequential) - log(t_cross_fresh)
```

Treat interpolation as an estimate within a five-update interval, not exact optimizer-step resolution. Also retain raw brackets and repeat the decision using observed upper-bracket crossings. If the sensitivity result changes the decision, classify the slowdown as inconclusive.

Within each paired run, average log-ratios across prespecified probe domains, then form the paired-run confidence interval. A plasticity signal requires exp(mean_log_ratio) >= 1.25 and the lower confidence bound on mean_log_ratio > 0. Threshold-at-initialization and unreached targets are uninformative/censored; never insert epsilon step counts or substitute the 2,000-update limit as an observed crossing. If fewer than 80% of prespecified pairs are uncensored, the ratio gate is inconclusive. Report censoring patterns and normalized learning-curve area for every probe, including censored ones. The uncensored estimate is conditional on its eligible subset; do not generalize it to censored probes.

| Recovery pattern | Decision |
|---|---|
| Adapter only restores | Adapter-related impairment is plausible; continue attribution checks |
| Decoder only restores | Stop adapter-plasticity claim |
| Both independently restore | Attribution ambiguous; examine joint and optimizer-only resets |
| Neither independently restores | Examine joint reset and reset completeness |
| Optimizer-only restores | Optimizer history is a competing explanation; compare common reset policies |

Judge recovery using learning trajectories, not immediate post-reset predictions.

## 20. Metrics

Let M[t,d] be test mIoU in percentage points on domain d after training stage t.

Report current-domain mIoU, per-class IoU, learning-curve area, valid-pixel accuracy, macro Dice, stage-by-domain matrix, retention, forgetting, backward transfer, runtime, and resources.

For T distinct domains:

```text
Retention R = mean(M[T,d] for d = 1,...,T-1)

Forgetting = mean(
    max(M[t,d] for t = d,...,T-1) - M[T,d]
    for d = 1,...,T-1
)

Late adaptation A = mean(M[t,t] for t in the final half of the sequence)
```

Fix the final-half index set before confirmation. Report recurring-domain metrics separately, including time since last exposure.

Aggregate confusion matrices within each domain before computing mIoU. False positives count even when that class is absent in ground truth. Exclude a class from domain macro IoU only when union is zero. Report class support and a fixed-class sensitivity analysis. Give domains equal weight in retention. Pixels are not independent statistical replicates.

## 21. Statistical protocol

### Locked primary comparison

**Quantum candidate:** the §2-9 model, 16 x 16 grid, frozen PCA, 128 MiB feature replay, and the §15 DER++-style segmentation loss. **Comparator:** the strongest development-selected classical competitor across both §17 tiers, using the same 128 MiB record budget and DER++-style loss. Its hyperparameters may be tuned with equal opportunity. ER-only, 32 MiB, zero replay, and other circuit resolutions are secondary comparisons and cannot replace this primary comparison after seeing results.

Select the comparator by highest mean final all-seen-domain development mIoU after 2,000 updates/domain; break numerical ties within 1e-6 mIoU point by lower measured total training time, then stable configuration ID. Evaluate only completed members of the predeclared full roster. If that roster cannot be completed, state that comparator selection is unfinished and do not run a confirmatory gate. Freeze the comparator configuration, roster, scores, and sample size before accessing confirmation outcomes.

- Five paired development runs estimate cost and variability.
- Plan at least ten confirmatory pairs.
- Determine final sample size before confirmation from pilot uncertainty for both adaptation and retention.
- Primary paired test: Student t test on run-level mean differences, with two-sided 95% Student-t confidence intervals; for retention test the shifted difference against -1 point using the corresponding lower bound.
- Add paired-bootstrap sensitivity analysis.
- Do not switch tests according to which passes.
- Apply Holm correction to secondary superiority comparisons.

Inspect development difference distributions with Q-Q plots and outlier summaries. Five pilot pairs cannot reliably establish normality. Do not use a Shapiro-Wilk result to switch automatically to Wilcoxon, which has different assumptions and an estimand that need not be the mean difference. Report paired-bootstrap sensitivity with 10,000 run-pair resamples and a saved seed. Material disagreement or dominant outliers make the result inconclusive under this version; a redesigned primary analysis must be declared before fresh confirmation data, not selected post hoc. The joint primary success decision requires both conditions below, while additional superiority claims form the Holm-corrected secondary family.

Success requires both:

```text
observed_mean_adaptation_gain >= 2 mIoU points
AND lower_95%_CI(adaptation_gain) > 0

lower_95%_CI(retention_difference) > -1 mIoU point
```

This does not establish that the true adaptation gain is at least two points. The retention margin represents tolerable degradation; variance informs sample size, not the acceptable-loss margin.

Pair domain order, minibatch sampling, reservoir decisions, augmentation schedules, and shared-module initialization where shapes permit. Conclusions are conditional on the selected data/domain construction; seeds do not replace independent datasets.

## 22. Feasibility gates

| Gate | Requirement | Failure action |
|---|---|---|
| Data | Valid labels, usable domains, spatial isolation | Resolve data issues |
| Numerics | Finite outputs/gradients; SAM unchanged | Fix implementation |
| Tiny overfit | Strong fit on fixed small subset | Investigate data/model/loss |
| Supervised competence anchor | Completed U-Net reference and per-class headroom report under §22.1 | Gate-bearing competence decision remains pending if reference is incomplete |
| Initial competence | Within proposed two-point margin of competent SAM classical reference | Stop candidate |
| Branch dependence | Branch removal costs >=1 validation mIoU point | Stop functioning candidate after correctness checks |
| Learned contribution | Trainable circuit improves over frozen circuit | Reject learned-circuit claim otherwise |
| Plasticity | Demonstrable new-domain learning slowdown | Stop plasticity-rescue claim otherwise |
| Attribution | Adapter involvement survives reset checks | Narrow/reject mechanism otherwise |
| Practical comparison | Pass adaptation and retention rules | Stop quantum direction otherwise |
| Cost | Study fits measured budget | Revise declared scope before confirmation |

Branch dependence is a screening heuristic, not proof of benefit. Candidate rejection does not rule out all quantum adapters. Do not change decoder, capacity, thresholds, or scale after inspecting confirmatory results.

### 22.1 Fully supervised competence anchor

Add a standalone U-Net-style supervised reference on the identical first-domain training/calibration/diagnostic/test split. This reference is an external competence anchor, not a replacement encoder for the SAM-quantum model and not an additional continual-learning candidate.

- Randomly initialized encoder channels 32, 64, 128, 256, bottleneck 512.
- Two bias-free 3x3 convolution -> GroupNorm(8 groups, eps=1e-5) -> ReLU operations per level; 2x2 max pooling between encoder levels.
- Four decoder levels: bilinear upsample by two, concatenate corresponding encoder skip, then two convolution/GroupNorm/ReLU operations. Output channels 256, 128, 64, 32 respectively.
- Final biased 1x1 convolution to eight logits at input resolution. Same valid pixels, class vocabulary, 1024 input tiles, and augmentation views.
- Train all reference weights with the same CE + 0.5 Dice loss. Adam, lr=1e-3, betas=(0.9,0.999), eps=1e-8; no replay and no future-domain data. Batch one with four-step accumulation to target effective batch four. Use FP32 first; record any mixed-precision execution separately.
- Up to 10,000 optimizer updates, evaluating calibration every 250. After at least 2,000 updates, stop on 2,000 updates without a >=0.5-point improvement over the running best calibration mIoU. Select the best calibration checkpoint. If the budget ends while the reference is still materially improving, mark the anchor undertrained and competence undecided rather than treating it as a ceiling.
- Use three development seeds for the anchor. Report individual and mean absolute mIoU, all per-class IoUs, valid-pixel accuracy, learning curves, and SAM-minus-U-Net gaps, particularly road and building. Include a training-majority-class predictor as a sanity anchor.

The competence decision must report absolute scores and the reference gaps, not just quantum-minus-classical-SAM differences. Proposed practical thresholds for the first-domain diagnostic mean: SAM quantum and its classical SAM reference must each exceed the majority predictor by at least 10 mIoU points, remain within five mIoU points of the completed U-Net anchor overall, and within ten IoU points for road/building where ground-truth support exists. These are protocol tolerances, not expected dataset results. Missing road/building support is reported as an unevaluable class anchor and prevents claims about those classes. Retain the separate two-point quantum-versus-SAM-classical competence margin. Failure/incomplete training stops or leaves pending the competence gate; do not weaken margins after observing results.

A supervised reference is not a mathematical upper bound. Differences include trainable representation, decoder, and training budget; use the anchor to expose headroom, not to attribute it solely to SAM stride.

## 23. PennyLane execution and profiling

Use a PyTorch-interface QNode. Begin with analytic expectations. Use FP64 for correctness tests; adopt FP32/complex64 only after reference agreement checks.

Benchmark compatible configurations:

1. `default.qubit`, analytic backpropagation.
2. `lightning.qubit`, adjoint differentiation.
3. `lightning.gpu`, if installed and compatible.

Test circuit-parameter gradients and input-angle gradients for trainable-projection ablations. Do not assume CUDA tensors select a GPU simulator. Do not assume `batch_input` supports trainable encoded inputs.

Flatten locations to `[B*g*g,8]`. Start with 64-instance chunks, then profile 128 and 256. Autograd may retain intermediate states across chunks; measure complete backward peak memory.

Micro-benchmark:

- Ten warm-up iterations.
- Fifty measured iterations.
- CUDA synchronization around measurements.
- Median and upper-percentile latency.
- Peak memory and CPU-GPU transfer costs.
- Prediction and gradient agreement with reference.

Report simulator state/density-matrix storage, actual peak CPU RSS and CUDA allocation, differentiation overhead, circuit executions, shot counts, and seconds per complete optimizer update separately from SAM extraction and the decoder. Theoretical state size alone is not measured training memory. CPU execution must not disappear from resource accounting simply because it uses zero GPU memory.

Only build a custom simulator after profiling establishes a substantial bottleneck. Source: [PennyLane differentiation support](https://docs.pennylane.ai/en/stable/introduction/interfaces.html).

## 24. Finite shots and synthetic noise

Secondary only, after analytic feasibility.

Shots: 256, 1,024, 4,096.

For 32 rotation parameters, basic parameter-shift uses approximately 64 shifted evaluations/input plus forward evaluation. With four examples and a 16 x 16 grid, there are 1,024 encoded instances/minibatch, or approximately 66,560 forward-plus-shift executions before multiplying by shots. Record actual backend counts.

Synthetic noise:

- Local depolarizing channel after each gate on participating wires.
- After CNOT, independent single-qubit channels on its two wires.
- p in {0, 1e-4, 1e-3}.
- No readout error in the main sweep.
- Readout error only as a separate ablation.

**Locked noise convention:** use PennyLane `qml.DepolarizingChannel(p)` with E(rho) = (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z). Thus p=3/4, not p=1, is fully depolarizing. Keep p fixed, not trainable. Apply after all encoding and trainable single-qubit gates; after each CNOT apply one independent channel to each participating qubit. Ideal X readout has no added measurement noise. Do not reinterpret p as average two-qubit gate infidelity.

**Device and gradients:** use `default.mixed`, eight wires, PyTorch interface. For analytic noisy training use `shots=None`, `diff_method="backprop"`; for finite-shot noisy training use `diff_method="parameter-shift"` with shots 256/1,024/4,096. Validate both on the pinned dependency version before a noisy run; no adjoint fallback for density-matrix noise is presumed. If the specified combination is unsupported, report the noise arm blocked rather than silently changing the model. Use FP64/complex128 correctness checks and benchmark smaller chunks before scaling density-matrix batches.

References: [channel convention](https://docs.pennylane.ai/en/stable/code/api/pennylane.DepolarizingChannel.html), [default.mixed device](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_mixed.DefaultMixed.html). These are synthetic sensitivity settings, not calibrated hardware error rates or values taken from the superconducting continual-learning study. Include noisy training, not only noisy inference, before discussing robustness.

## 25. Required verification

- Shapes and parameter counts.
- Frozen SAM unchanged after updates.
- PCA uses only authorized first-domain data.
- Angle bounds and zero-vector behavior.
- Circuit output bounds and signed-input behavior.
- Jacobian sensitivity across multiple inputs.
- Autograd/finite-difference gradient agreement.
- Batched/single-instance and chunk-size agreement.
- CPU/backend reference agreement.
- Nonzero circuit gradients through the residual path.
- Output-projection norm bound.
- Mask alignment, ignored pixels, missing classes, all-ignored samples.
- Hand-calculated metric cases.
- Replay capacity and deterministic reservoir selection.
- Historical-cache access restrictions.
- Checkpoint/restart equivalence.
- Diagnostic resets touch only intended state.

Use FP64 gradient comparison tolerances around 1e-6, with absolute tolerances near zero. Determine and record FP32 tolerances before performance runs.

## 26. Software interfaces

| Module | Responsibility |
|---|---|
| Data audit | Pairing, labels, groups, domains, leakage checks |
| SAM extractor | Normalization, frozen encoding, caching |
| PCA fitter | Training-only sampling and fitted buffers |
| Adapter library | Interchangeable quantum/classical transforms |
| Decoder | Dense semantic logits |
| Replay manager | Byte limits, reservoir state, detached targets |
| Trainer | Sequential updates/checkpoints |
| Diagnostics | Reset probes and fresh references |
| Evaluator | Domain/scene metrics without training access |
| Profiler | Time, memory, circuit and shot counts |
| Statistical analysis | Paired effects, intervals, corrections |

```python
features = encoder(images)             # [B,256,64,64]
adapted = adapter(features)             # [B,256,64,64]
coarse_logits = decoder(adapted)        # [B,8,64,64]
logits = resize_logits(coarse_logits)   # [B,8,1024,1024]
```

## 27. Checkpoints and experiment outputs

Save adapter/decoder weights, encoder checksum, PCA buffers, alpha, optimizer states, replay and reservoir state, all RNG states, domain/update position, resolved configuration, manifest hashes, and environment versions.

Required outputs:

- Dataset audit and partition manifests.
- Model/parameter summary.
- Resolved configuration and dependency versions.
- Training/validation histories in CSV or JSON.
- Stage-by-domain mIoU matrix and per-class metrics.
- Fresh/reset diagnostic learning curves.
- Circuit/gradient/residual diagnostics.
- Replay byte ledger and sample identifiers.
- Resource profiling results.
- Paired statistical results and confidence intervals.
- Prediction, ground-truth, and error visualizations.
- Explicit gate-by-gate pass/fail/inconclusive report.

Do not save plots without their underlying numerical data.

## 28. Budget and items resolved by feasibility

The initial allocation is capped at **48 GPU-hours**. Record CPU-only circuit computation in CPU-hours and wall-clock time separately. This cap covers a staged pilot, not the entire control matrix or confirmatory study.

### Staged hard allocations and hyperparameter cost forecast

| Stage | Maximum GPU-hours | Deliverable |
|---|---:|---|
| Stage 0: audit, numerics, tiny overfit, profiling, pilot-only cache | 8 | Correctness report, feasible split branch, measured cost model; no scientific gate pass |
| Stage 1: supervised anchor and first-domain SAM competence | 16 | Absolute/per-class competence report, or explicit pending/stop |
| Stage 2: locked development roster and full-budget comparisons | 16 | Completed comparator selection if affordable; otherwise incomplete, no winner claim |
| Stage 3: calibration references, fine-resolution reset/plasticity probes | 8 | Diagnostic report or explicit inconclusive/budget stop |
| **Total** | **48** | Feasibility only |

Do not borrow an exhausted stage's allowance from later stages. Unspent hours stay unspent unless a revised budget is declared before affected outcomes are inspected. Finite-shot/noise experiments and confirmation require separate budgets after feasibility.

Before Stage 1 and before committing any hyperparameter matrix, save a cost ledger using measured per-model update time and evaluation time:

```text
forecast_hours = 1.25 * (
    extraction_seconds
    + sum_over_runs(domains * (updates * seconds_per_update
                              + evaluations_per_domain * seconds_per_evaluation))
    + anchor_seconds + fresh_reference_seconds + reset_probe_seconds
) / 3600
```

Use 2,000 updates/domain and 401 evaluations/probe in the diagnostic forecast, plus five calibration seeds, all reset arms, folds if selected, and checkpoint/I/O overhead. CPU wall time is forecast separately with the same accounting. Count GPU allocation time even when a CPU circuit leaves the GPU waiting.

The 16-configurations-per-family number is a ceiling; at 2,000 updates and multiple seeds it may be unaffordable. Fix an equal, smaller candidate count from costs alone before comparing scores, or report that the complete development comparison requires more resources. Do not retain 200-update-selected winners as substitutes. Never claim that a budget-limited subset establishes the strongest classical comparator. A partial pilot may validly finish with Stage 0 only; it does not establish scientific failure or success.

Before full launch, determine:

1. Qualifying domains and source-scene counts.
2. First-domain classical reference competence.
3. Circuit and full-training-step throughput.
4. Peak training memory.
5. Whether the adapter contributes meaningfully.
6. Whether measurable plasticity loss exists.
7. Pilot variability and confirmatory sample size.
8. Whether total study cost fits available resources.

A source commit and checkpoint hash must be recorded when dependencies are installed. Do not invent measured runtime, benchmark accuracy, dependency pins, or final domain identities in advance.

## 29. Review summary

The primary candidate is a frozen SAM ViT-B encoder with 1024 x 1024 RGB input, 256 x 64 x 64 encoder output, a 16 x 16 spatial eight-qubit residual adapter with 32 trainable circuit angles, and a three-convolution semantic decoder. It has **371,624 trainable parameters** and produces **eight-class 1024 x 1024 logits**.

The empirical sequence is: competent segmentation -> adapter dependence and learned contribution -> measurable plasticity impairment -> attribution -> stronger adaptation with non-inferior retention against practical classical alternatives.

The study stops if classical alternatives match the circuit or if the SAM learner does not exhibit the hypothesized adapter-related impairment. Simulation results are not evidence of quantum computational advantage.
