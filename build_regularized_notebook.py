"""Derive an opt-in tuning notebook without changing the baseline or training code."""
import ast
import json
from pathlib import Path

root = Path(__file__).parent
nb = json.loads((root / 'Run_Full_Project_A100.ipynb').read_text(encoding='utf-8'))
for cell in nb['cells']:
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    if cell['cell_type'] == 'code':
        source = source.replace('RUN_NAME = "a100_quantum_seed42"',
            'ADAPTER = "quantum"  # quantum, classical, or none; compare separate matched runs\n'
            'RUN_NAME = f"a100_{ADAPTER}_regularized_20ep_v1_seed42"')
        source = source.replace('MAX_UPDATES = 2000  # Per continual domain',
            'MAX_UPDATES = 1_000_000_000  # Epoch-controlled run; update cap is effectively inactive')
        source = source.replace('EPOCHS = 200',
            'EPOCHS = 20  # Initial experiment: 20 epochs per region\n'
            'LEARNING_RATE = 0.0003\nQUANTUM_LEARNING_RATE = 0.003')
        source = source.replace('"dataset": {"manifest": str(MANIFESTS[dataset]), "split_source": SPLIT_SOURCES[dataset],',
            '"dataset": {"augment": True, "manifest": str(MANIFESTS[dataset]), "split_source": SPLIT_SOURCES[dataset],')
        source = source.replace('"checkpoint": str(SAM), "adapter": "quantum",',
                                '"checkpoint": str(SAM), "adapter": ADAPTER,')
        source = source.replace('"cache_dir": str(REPO_DIR / "cache" / "sam_vit_b" / dataset)',
                                '"cache_dir": None  # Required for online image augmentation')
        # Keep the closing dictionary brace on its own line, outside the comment.
        source = source.replace('None  # Required for online image augmentation},',
                                'None},  # Required for online image augmentation')
        source = source.replace('"training": {"epochs": EPOCHS, "max_updates_per_domain": MAX_UPDATES}',
            '"training": {"epochs": EPOCHS, "max_updates_per_domain": MAX_UPDATES,\n'
            '                     "learning_rate": LEARNING_RATE, "quantum_learning_rate": QUANTUM_LEARNING_RATE}')
        source = source.replace('if RUN_RANK_CHECK:', 'if RUN_RANK_CHECK and ADAPTER == "quantum":')
        ast.parse(source)
    else:
        source = source.replace('# QCL Final Boss — full A100 server runner',
                                '# QCL Final Boss — A100 regularized experiment')
        source = source.replace('BF16 AMP, frozen SAM feature caching, and batch size 2 are enabled. Tune batch/chunk sizes after Stage 0 if needed;',
            'BF16 AMP and batch size 2 are enabled. Random horizontal/vertical flips are applied jointly to training images and masks.\n'
            'Feature caching is disabled so SAM encodes the actual augmented images. Validation is unaugmented. Tune batch/chunk sizes after Stage 0 if needed;')
        source = source.replace('The defaults retain the full 2,000-update/domain budget and 200-epoch ceiling. The epoch ceiling can stop a run before its update budget.',
            'This exploratory run uses 20 epochs per domain, with an effectively inactive update cap.\n'
            'It is not a fixed 2,000-update research comparison. Small domains receive fewer updates. Increasing the epoch limit later requires a new run under the current strict resume checks.')
        source = source.replace('- **Slow first epoch:** SAM features are cached and PCA is initialized using training samples. Leave enough disk space for the cache.',
            '- **Slower training:** augmented images pass through frozen SAM every batch; no feature cache is used. PCA still uses training samples only.')
        source = source.replace('- **Final status below 2,000 updates:** the epoch ceiling was reached. For a fresh full-budget experiment, raise `EPOCHS` with a new run name.',
            '- **Different update counts:** expected across domains with different sizes under the 20-epoch limit.')
    cell['source'] = source

intro = {
    'cell_type': 'markdown', 'id': 'regularization-notes', 'metadata': {},
    'source': '''## What changed and how to compare
- Start a **fresh model/run**; do not resume the baseline checkpoint into this configuration.
- Training flips enabled; original baseline had no augmentation.
- Classical/decoder learning rate: 0.001 → 0.0003. Quantum learning rate: 0.01 → 0.003.
- Maximum exposure per region: 200 → 20 epochs. The update cap is effectively disabled so it does not truncate the 20 epochs.
- Same seed, manifests, source groups, label mapping, and held-out split rules.
- Existing loss, replay, model architecture, and scoring formulas are unchanged.

These are tuning choices motivated by the observed Aachen plateau, not a demonstrated score improvement.
Lower learning rates plus shorter training can underfit some regions; assess validation metrics at each domain endpoint.
Because Aachen validation informed these settings, improvement there is development evidence, not independent confirmation.

Run `ADAPTER="quantum"` first. For matched controls, run again with `ADAPTER="classical"` or `"none"`;
the run names differ automatically. Compare validation at the **same domain endpoint** and with identical splits/budgets.
Current-domain online training metrics and all-seen validation metrics have different scopes after the first region.
Use validation to choose settings; reserve the held-out test for the final selected configuration.
The evaluation cell below is disabled by default until you finish choosing settings.
'''
}
nb['cells'].insert(1, intro)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'REPORTS = {}' in cell['source']:
        cell['source'] = cell['source'].replace('REPORTS = {}', 'EVALUATE_HELD_OUT = False  # Set True only for the final selected experiment\nREPORTS = {}')
        cell['source'] = cell['source'].replace('    # A new report directory',
            '    if not EVALUATE_HELD_OUT:\n        print("Training complete; use validation reports to choose settings before testing.")\n        continue\n    # A new report directory')
    if cell['cell_type'] == 'code' and 'from IPython.display' in cell['source']:
        cell['source'] = cell['source'].replace('REPORTS[dataset] / "metrics.json"', 'OUTPUTS[dataset] / "continual_metrics.json"')
        cell['source'] = cell['source'].replace('list(REPORTS[dataset].rglob("*.png"))[:6]', 'list(OUTPUTS[dataset].glob("*.png"))[:6]')
        cell['source'] = cell['source'].replace('        print(path.read_text()[:16000])', '        print(path.read_text()[:16000] if path.exists() else "Not yet available")')
    if cell['cell_type'] == 'code':
        ast.parse(cell['source'])
path = root / 'Run_Regularized_Project_A100.ipynb'
path.write_text(json.dumps(nb, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(path)
