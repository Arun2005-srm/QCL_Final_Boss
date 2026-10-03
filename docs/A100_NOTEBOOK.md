# Run the regularized experiment on A100

## Update the server

Stop the old training process before running the new experiment on the same GPU.

```bash
cd /raid/workspace/QCL_Final_Boss
git pull --ff-only origin main
```

If Git reports conflicting local changes or untracked notebooks, preserve those files before pulling. Do not discard your server edits or outputs.
For a new checkout:

```bash
cd /raid/workspace
git clone https://github.com/Arun2005-srm/QCL_Final_Boss.git
```

Open `Run_Regularized_Project_A100.ipynb` in Jupyter with a Python 3.10–3.12 kernel.
Run cells from the top. Dependencies are installed into the project's `.venv-a100`; project subprocesses use that environment.

## Settings

| Setting | Regularized experiment |
| --- | --- |
| Project | `/raid/workspace/QCL_Final_Boss` |
| Datasets | OpenEarthMap, then LandCover.ai as a separate model |
| Epochs | 20 per region, not 20 across the complete dataset |
| Update cap | Effectively inactive; epochs determine the budget |
| Training augmentation | Paired horizontal and vertical flips |
| Decoder/classical learning rate | 0.0003 |
| Quantum learning rate | 0.003 |
| Batch size | 2 |
| Feature cache | Disabled because augmented images must pass through SAM |
| Held-out evaluation | Off until settings are selected using validation |

Default data manifests are `/raid/workspace/RS_Dataset/openearthmap/pairs.csv` and
`/raid/workspace/RS_Dataset/landcover_ai/pairs.csv`. Verify column mappings and raw label meanings in Settings.
Existing weights are located under `/raid/workspace/AI4CV/models` or `/raid/workspace/sam_tests/SAM_test/checkpoints`;
the notebook can also download original SAM ViT-B weights.

For generated OpenEarthMap splits, Section 4b creates a separate manifest containing only regions with at least seven source groups.
In the inspected dataset, this excludes Bogota (5 samples), Kampala (5), and Kinshasa (3).
It preserves source group IDs and original files, records the exclusions, and adds `_eligible7` to the run name.
Existing manifest split assignments are preserved instead of applying this filter.

The default regularized output is `outputs/a100_quantum_regularized_20ep_v1_seed42_eligible7/`.
The earlier baseline is preserved. Never resume its checkpoint into this changed configuration.

## Compare and extend

Use `history.json`, `training_curves.png`, and domain-endpoint validation in `continual_metrics.json` to assess the experiment.
After the first region, training metrics describe the current region while validation aggregates all seen regions.
Compare the same domain endpoint and split across runs. A higher training score alone is not an improvement.
The inspected baseline showed overfitting on Aachen, but this configuration has not yet been evaluated on A100.
Lower learning rates and shorter exposure may underfit some regions; use validation to decide the next experiment.

For a matched control, set `ADAPTER` to `classical` or `none`; its output name changes automatically.
When selecting the final configuration, set `EVALUATE_HELD_OUT=True` in Section 7 and rerun that cell.
Completed training is reused and a new test report is written. Avoid repeatedly choosing settings on the test split.

Interrupted training can resume with exactly the same settings at epoch boundaries. With online augmentation,
worker RNG state is not fully preserved, so bit-for-bit equivalence to uninterrupted training is not guaranteed.
Increasing `EPOCHS` after this run currently requires a new `RUN_NAME` and fresh training; the existing engine rejects changed configs on resume.
A completed stream cannot simply resume each previous region for extra epochs without defining a new training schedule.

## Notebook maintenance

Regenerate notebooks after changing their generators:

```bash
python build_a100_notebook.py
python build_regularized_notebook.py
```

The generators require only Python's standard library and check each code cell's syntax.
Notebook schema validation was performed locally; real GPU execution and dataset audits run on the server.
