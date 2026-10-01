# SAM-based quantum continual segmentation: pre-run manifest

Version: 1.0  
Date: 2026-10-01  
Status: execution decisions recorded; measured fields pending. Gate-bearing runs are not released.

## 1. Authority and review disposition

The operative architecture and scientific protocol remain **SAM_Quantum_Continual_Learning_Implementation.md, Version 1.1**. Its SHA-256 at creation of this manifest is:

`183065A43E881CC9E1EEFE14BCAF8E910A9CF30C45509324212992F5B3F8D382`

This manifest resolves the execution choices raised in the two final reviews. It does not replace the implementation specification. Archive both files with every run. Any later amendment needs a dated change log stating whether affected outcomes have been inspected.

Both final reviews accept the specification for Stage 0. That is approval of the feasibility design, not evidence of accuracy, trainability, affordable throughput, or quantum benefit. This document records decisions; it does not claim that experiments have run.

The scientific question remains whether a quantum-circuit adapter improves measured late adaptation while preserving retention relative to strong classical alternatives in this SAM learner. A simulated eight-qubit model cannot establish quantum computational advantage.

## 2. Model identity remains fixed

| Item | Locked setting |
|---|---|
| Encoder | Original SAM ViT-B image encoder; frozen; checkpoint `sam_vit_b_01ec64.pth` |
| Input / encoder output | RGB 1024 × 1024 / 256 × 64 × 64 |
| SAM blocks | 12 transformer blocks; prompt encoder and native mask decoder excluded |
| Adapter grid | 16 × 16 locations |
| Projection | Frozen first-domain PCA, 256 → 8, followed by the specified direction normalization |
| Circuit | Eight qubits; RY encoding; two RX-then-RY trainable blocks with CNOT chains; eight X readouts; 32 trainable angles |
| Output projection | 8 → 256, no bias, spectral norm at most one |
| Residual scale | Fixed after the specified calibration; no increase to rescue a failed gate |
| Decoder | Specified three-convolution semantic decoder, 256 → 128 → 64 → 8 |
| Prediction | Eight-class logits at 1024 × 1024 |
| Trainable parameters | 371,624 under the specified primary architecture |
| Primary replay | 128 MiB, fixed-record accounting, DER++-style loss |

No DINO component is introduced. The full layer, initialization, loss, optimization, and preprocessing definitions remain in v1.1.

## 3. Primary comparator and adaptation sensitivity

Keep the primary comparator selection unchanged: highest mean final all-seen-domain development mIoU across the complete, predeclared classical roster in both tiers, with the v1.1 tie rules. Both primary arms use 128 MiB replay and the specified DER++-style loss, with 2,000 updates per domain.

Add one secondary sensitivity comparison: select the classical configuration with the highest development late-adaptation score, using the same final-half-domain metric as the primary adaptation endpoint. Select from the same completed roster and use the same tie rules. Freeze both comparator identities before confirmation outcomes are accessed.

Report quantum-minus-classical adaptation and retention differences against both comparators. The sensitivity result cannot replace or rescue the primary comparison. If the selected configurations coincide, report that fact and do not count the duplicate as an independent comparison. Include any additional superiority claim in the prespecified Holm-corrected secondary family.

Pending before comparator search: enumerate every family, configuration ID, search range, paired seed, and domain order in the machine-readable run manifest. The specification's candidate-count ceiling is not a promise that the entire search is affordable.

## 4. Replay in plasticity probes

**Decision: no replay in any of the six probe copies.** This applies to unchanged sequential, fresh, adapter reset, decoder reset, optimizer reset, and joint reset copies. Fresh-reference calibration runs used to define probe targets also use no replay.

The sequential checkpoint being diagnosed still comes from the prescribed continual training procedure with replay. At the probe fork, preserve its specified weights and optimizer state, but disable replay sampling and replay losses in all probe arms. Probe copies never write back to the main training stream or replay store.

Use matched current-domain example order, augmentation seeds, sample exposure, update count, and evaluation cadence across the six copies. Apply exactly the reset semantics in v1.1; disabling replay is not an instruction to reset every optimizer.

Interpret these probes as current-domain plasticity under matched no-replay adaptation. They do not directly estimate plasticity while the full DER++ objective remains active. A replay-enabled probe would be a separately declared secondary experiment.

Preserve the 2,000-update budget, evaluation at zero and every five updates, frozen calibration-derived targets, sustained crossings, interpolation, log-ratios, censoring rules, and upper-bracket sensitivity check. Small diagnostic partitions may make this result inconclusive; finer evaluation does not create additional independent images.

## 5. Noise placement and memory

Interpret the v1.1 phrase about single-qubit gates as **after each gate**, not once after a group of gates:

1. Apply one depolarizing channel after each RY encoding gate.
2. Apply one depolarizing channel after each trainable RX and each trainable RY gate.
3. After each CNOT, apply one independent channel to each participating wire.
4. Add no measurement-noise channel to the ideal X expectation readout.

This gives 8 encoding-channel placements, 32 trainable-single-qubit placements, and 28 placements after 14 CNOTs: **68 single-qubit channel placements per circuit instance**.

Retain the declared convention:

`E(rho) = (1-p) rho + (p/3) (X rho X + Y rho Y + Z rho Z)`.

Use fixed p in {0, 0.0001, 0.001}. These are synthetic sensitivity settings, not hardware-calibrated gate errors. Retain `default.mixed`, the PyTorch interface, analytic `shots=None` with backprop, and the separately budgeted finite-shot parameter-shift arms. Actual support must pass the pinned-environment smoke tests before use.

One eight-qubit density matrix has 256 × 256 complex entries. At complex128 its raw storage is 1,048,576 bytes, or 1 MiB. This excludes saved differentiation intermediates, framework allocations, gradients, and copies. Do not infer training memory from raw-state size alone.

Start profiling with one circuit instance, then chunks of two and four; grow only within measured memory limits. Record CPU peak resident memory as well as accelerator memory and actual device placement. Do not default to a 512-instance density-matrix backward pass.

### Exact gradient accumulation requirement

Chunking must preserve the full segmentation loss. The spatial decoder and Dice loss couple locations; summing independently computed chunk-local segmentation losses is not equivalent.

For analytic circuit training, an implementation may use this two-pass vector-Jacobian procedure:

1. Compute all circuit outputs in small chunks without retaining their circuit graphs.
2. Assemble the output grid as a detached leaf requiring gradients. Run the output projection, spatial interpolation, decoder, and full loss once. Backpropagate to obtain classical parameter gradients and the full gradient with respect to circuit outputs.
3. Recompute each circuit chunk with the unchanged circuit parameters and backpropagate its saved output gradient. Accumulate circuit parameter gradients across chunks.
4. Apply one optimizer step after all gradients are accumulated, followed by the specified output-projection constraint.

The PCA/input front end is frozen in this protocol. Validate this method against an unchunked analytic tiny case, comparing the loss and every trainable parameter gradient. Save absolute and relative errors and predeclared numerical tolerances. Finite-shot recomputation requires separate estimator validation; analytic equivalence does not establish stochastic equivalence.

Noise experiments remain secondary and separately budgeted. Failure of their backend smoke test does not silently change the primary noiseless model.

## 6. PCA bottleneck and supervised projection suggestion

Keep frozen eight-component PCA in the primary model. The unconstrained classical adapter tier already tests whether the constrained front end is practically competitive. A poor result cannot be rescued by silently replacing PCA.

Defer the suggested PCA-versus-LDA experiment from the mandatory 48-hour roster. Record it as an optional, separately budgeted development diagnostic, declared before its outcomes are observed. This avoids adding an uncosted family to an already tight feasibility study.

If activated, use the same first-domain training-only features and labels for fitting, the same downstream classical control, the same sample budget, and the same evaluation split. Standard LDA with eight classes has at most seven discriminant directions, potentially fewer with missing classes. Therefore specify a rank-matched PCA comparison; do not describe it as an interchangeable eight-dimensional projection without defining padding and controls. Freeze its complete protocol before running it.

A supervised projection benefit would motivate a new architecture study. It would not prove that PCA caused all failures or retroactively change the primary experiment.

## 7. Rank-regularizer source contingency

Before claiming a faithful L2-ER port, pin the source repository and commit, record the license, and verify the implementation against the source on fixed small activation matrices. Compare both loss and gradients, including the source's update order and rank-spectrum convention.

If source verification is unavailable or fails:

- Mark that comparator unavailable or unverified, including the exact reason.
- Continue unrelated Stage 0 checks and descriptive experiments that remain valid.
- Report the existing L2-only baseline under its own name; it is not a substitute for the missing rank-regularized family.
- Keep complete-roster comparator selection unfinished. Do not release the confirmatory gate or claim victory over the strongest prescribed classical competitor.

This preserves v1.1's full-roster rule. A reduced-roster study requires an explicit prospective protocol amendment and narrower claims.

## 8. Sample-size ceiling and resource decisions

Set a maximum of **30 independent paired confirmation runs**, with the existing minimum of ten. Determine the required count from development uncertainty for both adaptation and retention before confirmation. Record the power method, target power, assumed effects, variance estimates, and resulting count; those numerical planning inputs remain pending and must not be chosen from confirmation results.

If the calculation requires more than 30 pairs, report the proposed confirmation as infeasible under this manifest. Do not truncate to 30 and call it adequately powered. Any larger study needs a prospective amendment and a separate resource allocation. Folds, domains, tiles, and repeated evaluations are not additional independent run pairs.

The 48 GPU-hour feasibility allocation remains:

| Stage | Cap | Required accounting |
|---|---:|---|
| 0 | 8 hours | Audit, numerical checks, tiny overfit, profiling, pilot cache |
| 1 | 16 hours | Supervised anchor and first-domain SAM competence |
| 2 | 16 hours | Complete development roster at 2,000 updates/domain |
| 3 | 8 hours | Fresh calibration references and six-arm plasticity diagnostics |

Confirmation, finite-shot studies, and noise studies require separate budgets. A reviewer's illustrative runtime is not a measurement. In particular, parameter-shift timing must not be substituted for the primary analytic-training timing.

Before Stage 1 and before search begins, forecast total cost from measured update and evaluation times. Include every family, five paired development runs, selected domain count, folds, five fresh calibration seeds, all six probe arms, 401 evaluations per probe, extraction, anchor training, checkpoints, and the v1.1 overhead allowance. Count allocated GPU time while CPU work leaves the GPU idle; report CPU wall time separately.

If the required workload does not fit, record a budget stop or declare a revised budget before inspecting affected performance outcomes. Do not reduce update budgets, drop required competitors, borrow stage hours, or use short-run rankings to manufacture a completed comparison.

## 9. Anchor and residual interpretation

The supervised U-Net remains an empirical competence reference using the same split, valid labels, 1024 input tiles, and augmentation views. It is not an upper bound or proof that the frozen-SAM representation retains fine structures. Report undertraining flags, class support, per-class gaps, and majority-predictor performance alongside overall mIoU.

Compute each seed's actual residual bound and initial scale from its calibration data. Log realized residual-to-feature RMS at initialization, every 50 updates, and each domain endpoint. The roughly 3% cap is an illustrative value, not a universal constant.

If a numerically correct adapter fails the declared branch-dependence gate at that scale, stop that candidate. Branch removal establishes dependence only; it does not establish superiority over a classical adapter. Raising alpha or changing the projection requires a new protocol, not a repair to this run.

## 10. Stage 0 evidence to populate

All entries below are **pending**, not passed by document review.

| Evidence | Required saved result |
|---|---|
| Environment | OS, Python, PyTorch, PennyLane and plugin versions; accelerator model, memory, driver; source commit and dependency lock |
| Encoder | Checkpoint hash, loading result, frozen/eval checks, exact feature shape and parameter count |
| Data | Dataset release/path, raw mask-ID histogram, verified label mapping, invalid/ignore counts, image-mask alignment |
| Isolation | Source-group definitions and overlaps, counts per region, dev/confirmation separation, partition hashes |
| Split branch | Deterministically selected v1.1 branch, selected regions, independent group counts, fold assignments if applicable |
| Projection | Fit sample IDs, PCA buffers/hash, eigenvalue summary, normalization checks; no future-domain fitting |
| Circuit | Gate/readout ordering, sign-sensitivity checks, finite outputs and gradients, seeded FP64 stacked-Jacobian singular values and ranks |
| Residual | Calibration sample IDs, alpha, r0, feature RMS, norm constraint, per-seed bound and realized ratio |
| Loss and replay | Label/ignore handling, aggregate confusion matrices, replay byte accounting and insertion semantics |
| Optimization | Correct parameter groups, constraint application, checkpoint-resume reproducibility |
| Tiny overfit | Curves and prediction sanity checks under the prescribed engineering budget; no scientific success claim |
| Profiling | Per-arm update/evaluation timing, warm-up and repetitions, memory peaks, device placement, throughput and chunk size |
| Noise smoke test | Pinned device/interface/gradient combinations, p=0 agreement check, exact channel placements, memory-safe chunk validation |
| Baseline sources | Rank-regularizer source verification and any other source-dependent implementation checks |
| Budget | Measured per-stage forecast, complete roster and search count feasibility, CPU and GPU ledgers |

Raw mask IDs must be verified before even tiny-overfit training. Lack of suitable independent groups blocks the data protocol rather than permitting tile leakage.

## 11. Release conditions and next deliverable

Stage 0 may gather the evidence above. Before any gate-bearing stage starts, complete and archive the data/split manifest, environment lock, model configuration, comparator roster, seeds/domain orders, evaluation rules, source-verification status, and measured resource forecast. Fill stage-specific statistical inputs before the corresponding analysis or confirmation. Pending fields must be explicit in every status report.

Use these outcome labels consistently:

- **Correctness failure:** fix a demonstrated implementation defect, log the change, and rerun affected checks.
- **Data infeasible:** the declared independent-group requirements cannot be met.
- **Resource incomplete:** the required experiment roster does not fit or was not completed; no scientific winner or failure claim follows.
- **Scientific stop:** a completed, valid experiment fails a predeclared gate.
- **Inconclusive:** censoring, uncertainty, or sensitivity checks prevent the declared conclusion.
- **Proceed:** all prerequisite evidence for the next stage is present and its resource allocation is recorded.

The next research deliverable is a Stage 0 evidence report with measured values and explicit unresolved items. Approval of these documents alone does not supply any of those measurements.
