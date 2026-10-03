"""Generate the standalone server notebook using only the standard library."""
import ast
import json
from pathlib import Path
import textwrap

cells = []
def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(s).strip()+'\n'))
def code(s):
    s = textwrap.dedent(s).strip()+'\n'
    ast.parse(s)
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s))

md('''
# QCL Final Boss — full A100 server runner

Run this notebook in **Jupyter on your Linux A100 server**, using Python 3.10–3.12 (3.11 recommended).
It runs the real project commands: install → tests → dataset audit → Stage 0 → training → held-out evaluation → reports.
Both OpenEarthMap and LandCover.ai run as **separate experiments** because their class vocabularies differ.
One GPU is used; this repository has no distributed training launcher.

## Before Run All
1. The project is configured for `/raid/workspace/QCL_Final_Boss`; it is cloned there if missing.
2. Data paths default to `/raid/workspace/RS_Dataset`, with existing weights under `/raid/workspace/AI4CV` or `/raid/workspace/sam_tests`.
   If your data lives elsewhere, edit the manifest and checkpoint paths below.
3. Check the dataset settings below. The directory listing reveals filenames, **not CSV contents or mask semantics**.
   Column mappings and raw label mappings must match your actual prepared datasets. The audit fails on incompatible data.

The notebook creates a dedicated `.venv-a100` and runs Python subprocesses there, so no notebook kernel restart is needed.
It does not upload your Windows folder automatically. Upload it first if you want local changes instead of GitHub's version.
Keep Jupyter's server/kernel running during training. For disconnect-resistant execution, run Jupyter inside your server's job allocation or tmux.
''')
code('''
from pathlib import Path
import os, sys, subprocess, json, csv, datetime, platform, shlex

SERVER_ROOT = Path("/raid/workspace")
REPO_DIR = SERVER_ROOT / "QCL_Final_Boss"
REPO_URL = "https://github.com/Arun2005-srm/QCL_Final_Boss.git"
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # Keep scheduler allocation if present
RUN_NAME = "a100_quantum_seed42"  # Keep unchanged to resume; change for a fresh experiment
DATASETS = ["openearthmap", "landcover_ai"]
BATCH_SIZE = 2  # Conservative starting point for either 40 GB or 80 GB A100
NUM_WORKERS = 4
MAX_UPDATES = 2000  # Per continual domain
EPOCHS = 200
RUN_RANK_CHECK = True  # Full circuit Jacobian check can take time
RESUME = True  # Resume last.pt when present; never silently replace a run

# Dedicated environment; CUDA 12.4 wheels need a compatible host NVIDIA driver.
TORCH_VERSION, VISION_VERSION = "2.5.1", "0.20.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Existing prepared RGB image + semantic mask manifests from structure.txt.
# Map canonical names to CSV headers if different, e.g. {"image": "image_path"}.
# Set default_domain only if your metadata defines a single domain.
SETTINGS = {
    "openearthmap": {
        "manifest": SERVER_ROOT / "RS_Dataset/openearthmap/pairs.csv",
        "columns": {}, "default_domain": "", "domain_from_group_prefix": False,
        "label_map": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7},
        "ignore_values": [0], "color_map": None,
        "raw_root": None, "groups_csv": None, "independent_scenes_verified": False,
    },
    "landcover_ai": {
        "manifest": SERVER_ROOT / "RS_Dataset/landcover_ai/pairs.csv",
        "columns": {}, "default_domain": "landcover_ai", "domain_from_group_prefix": False,
        "label_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        "ignore_values": [], "color_map": None,
        "raw_root": None, "groups_csv": None, "independent_scenes_verified": False,
    },
}
# If OEM masks are already zero-based 0..7, use {i: i for i in range(8)} and
# set ignore_values to the actual void ID(s), e.g. [255]. Do not infer this from filenames.
# Binary building masks require a separate two-class config, not the eight-class OEM preset.
''')
md('''
## 1. Locate the code and create command helpers
Existing checkouts are reused without pulling or replacing files. Logs are appended under `notebook_logs/<RUN_NAME>`.
''')
code('''
assert platform.system() == "Linux", "Run this notebook on the Linux A100 server."
assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Select a Python 3.10–3.12 notebook kernel."
SERVER_ROOT = SERVER_ROOT.expanduser().resolve()
REPO_DIR = REPO_DIR.expanduser().resolve()
if not REPO_DIR.exists():
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
for name in ["train.py", "testing.py", "stage0.py", "audit_dataset.py", "requirements.txt"]:
    assert (REPO_DIR / name).is_file(), f"Wrong REPO_DIR: missing {name} in {REPO_DIR}"
os.chdir(REPO_DIR)
ENV = os.environ.copy()
ENV.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
LOG_DIR = REPO_DIR / "notebook_logs" / RUN_NAME
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run(args, log="setup.log"):
    args = [str(a) for a in args]
    print("$", shlex.join(args), flush=True)
    with (LOG_DIR / log).open("a", encoding="utf-8") as handle:
        handle.write("\\n" + datetime.datetime.now().isoformat() + " " + shlex.join(args) + "\\n")
        process = subprocess.Popen(args, cwd=REPO_DIR, env=ENV, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            result = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if result:
            raise subprocess.CalledProcessError(result, args)

run(["nvidia-smi"])
PY = REPO_DIR / ".venv-a100/bin/python"
if not PY.exists():
    run([sys.executable, "-m", "venv", str(PY.parent.parent)])
print("Project:", REPO_DIR, "\\nEnvironment:", PY)
''')
md('''
## 2. Install dependencies and verify CUDA
The pinned torch/torchvision pair comes from the [official PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/).
If CUDA initialization fails, check your server's driver and GPU allocation; a notebook cannot install a host driver.
PennyLane uses the project's `default.qubit` Torch backprop implementation. Do not switch to `lightning.gpu`: this code rejects it.
''')
code('''
run([PY, "-m", "pip", "install", "--upgrade", "pip"])
run([PY, "-m", "pip", "install", f"torch=={TORCH_VERSION}", f"torchvision=={VISION_VERSION}",
     "--index-url", TORCH_INDEX])
constraints = LOG_DIR / "constraints.txt"
constraints.write_text(f"torch=={TORCH_VERSION}\\ntorchvision=={VISION_VERSION}\\npennylane==0.42.3\\n")
run([PY, "-m", "pip", "install", "-r", "requirements.txt", "-c", constraints])
run([PY, "-m", "pip", "check"])
run([PY, "-c", "import torch, torchvision, pennylane; "
     "print('Torch', torch.__version__, 'CUDA runtime', torch.version.cuda); "
     "assert torch.cuda.is_available(), 'CUDA unavailable: check driver / allocation'; "
     "print('GPU', torch.cuda.get_device_name(0)); "
     "print('VRAM GiB', torch.cuda.get_device_properties(0).total_memory / 2**30); "
     "x=torch.randn(128,128,device='cuda'); print('CUDA matmul', (x@x).mean().item()); "
     "assert torch.cuda.is_bf16_supported(), 'BF16 required for configured AMP'"])
freeze = subprocess.check_output([str(PY), "-m", "pip", "freeze"], text=True, env=ENV)
(LOG_DIR / "environment-lock.txt").write_text(freeze)
run([PY, "-m", "pytest", "-q"], "tests.log")
''')
md('''
## 3. Find or download the original SAM ViT-B weights
Reuses checkpoint paths present in your server listing. The fallback download is from
[Meta's official SAM checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints).
''')
code('''
import urllib.request
checkpoint_candidates = [
    SERVER_ROOT / "AI4CV/models/sam_vit_b_01ec64.pth",
    SERVER_ROOT / "sam_tests/SAM_test/checkpoints/sam_vit_b_01ec64.pth",
    REPO_DIR / "checkpoints/sam_vit_b_01ec64.pth",
]
SAM = next((p for p in checkpoint_candidates if p.is_file()), checkpoint_candidates[-1])
if not SAM.is_file():
    SAM.parent.mkdir(parents=True, exist_ok=True)
    temporary = SAM.with_suffix(".download")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", temporary)
    assert temporary.stat().st_size > 300_000_000, "Checkpoint download looks incomplete"
    temporary.replace(SAM)
print("SAM checkpoint:", SAM)
''')
md('''
## 4. Inspect existing manifests or prepare raw data
Existing CSVs need `id,image,mask,group` and domain metadata. `domain` may come from a `region/source` group prefix,
or an explicit `default_domain`. Relative image/mask paths resolve against the CSV directory.
The notebook preserves an existing `split` column (`train`, `val`, `test`) automatically.
Generated splits require at least seven independent source groups per domain.

If you need raw preparation, set `manifest=None`, `raw_root=Path('/actual/source')`, and `groups_csv` above.
Raw OEM must have `<region>/images` and `<region>/labels`; raw LandCover.ai needs `images` and `masks`.
The `RS_Dataset/openearthmap` folder in your listing has `masks`, so use its existing CSV rather than the raw OEM preparer.
Only set `independent_scenes_verified=True` after verifying source independence. Tiles from one scene must share a group.
Old `*_image.npy`, `*_mask.npy`, and `*_sam1024.npy` files are not this project's input/cache format.
''')
code('''
MANIFESTS, SPLIT_SOURCES = {}, {}
for dataset in DATASETS:
    setting = SETTINGS[dataset]
    manifest = setting["manifest"]
    if manifest is None:
        assert setting["raw_root"], f"Set raw_root for {dataset}"
        prepared = REPO_DIR / "data_local" / (dataset + "_notebook")
        manifest = prepared / "samples.csv"
        if not manifest.exists():
            args = [PY, "prepare_dataset.py", "--kind", dataset, "--root", setting["raw_root"],
                    "--output", prepared, "--tile-size", "1024"]
            if setting["groups_csv"]:
                args += ["--groups-csv", setting["groups_csv"]]
            else:
                assert setting["independent_scenes_verified"], "Supply source group metadata before preparation"
                args += ["--assume-independent-scenes"]
            run(args, dataset + "_prepare.log")
    manifest = Path(manifest).expanduser().resolve()
    assert manifest.is_file(), f"Missing {manifest}; set SERVER_ROOT or manifest path"
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        first = next(reader, None)
    print(dataset, "headers:", headers, "\\nFirst sample:", first)
    assert first, f"Empty manifest: {manifest}"
    required = [setting["columns"].get(k, k) for k in ["id", "image", "mask", "group"]]
    assert set(required) <= set(headers), f"Set SETTINGS[{dataset!r}]['columns']; missing {set(required)-set(headers)}"
    split_col = setting["columns"].get("split", "split")
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        has_splits = any((r.get(split_col) or "").strip() for r in csv.DictReader(f))
    MANIFESTS[dataset] = manifest
    SPLIT_SOURCES[dataset] = "manifest" if has_splits else "generated"
    # Show raw IDs/colors from one mask; the subsequent audit checks every mask.
    mask = Path(first[setting["columns"].get("mask", "mask")])
    if not mask.is_absolute():
        mask = manifest.parent / mask
    run([PY, "-c", "import numpy as np,sys; from PIL import Image; "
         "a=np.asarray(Image.open(sys.argv[1])); print('Mask shape:',a.shape); "
         "print('Raw values (first 30):', np.unique(a.reshape(-1,a.shape[-1]),axis=0)[:30] "
         "if a.ndim==3 else np.unique(a)[:30])", mask], dataset + "_inspect.log")
''')
md('''
## 4b. Select OpenEarthMap domains eligible for generated splits
Exclude regions with fewer than seven source groups from this experiment. In the supplied counts these are
Bogota (5), Kampala (5), and Kinshasa (3): 13 samples. Original files and group identities are preserved.
The derived manifest and exclusion report are saved under `data_local/oem_eligible7`.
Results describe the retained regions only. Existing manifest splits are preserved without this filter.
''')
code('''
if "openearthmap" in DATASETS and SPLIT_SOURCES["openearthmap"] == "generated":
    helper = r"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict, Counter
from qcl.config import load_config, merge
from qcl.data import discover
cfg = merge(load_config('configs/datasets/openearthmap.yaml'), {'dataset': json.loads(sys.argv[1])})
records = discover(cfg)
groups = defaultdict(set)
for r in records:
    groups[r.domain].add(r.group)
excluded = {d for d, g in groups.items() if len(g) < 7}
kept = [r for r in records if r.domain not in excluded]
assert kept, 'No eligible domains remain'
out = Path('data_local/oem_eligible7')
out.mkdir(parents=True, exist_ok=True)
with (out / 'samples.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id','image','mask','group','domain'])
    writer.writeheader()
    writer.writerows({k: getattr(r, k) for k in writer.fieldnames} for r in kept)
report = {'source_manifest': cfg['dataset']['manifest'], 'minimum_groups': 7,
          'excluded_groups': {d: len(groups[d]) for d in sorted(excluded)},
          'excluded_samples': dict(Counter(r.domain for r in records if r.domain in excluded)),
          'retained_samples': len(kept), 'retained_domains': len({r.domain for r in kept})}
(out / 'exclusions.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
"""
    setting = SETTINGS["openearthmap"]
    dataset_settings = {k: setting[k] for k in ["columns", "default_domain", "domain_from_group_prefix"]}
    dataset_settings.update(manifest=str(MANIFESTS["openearthmap"]), split_source="generated")
    run([PY, "-c", helper, json.dumps(dataset_settings)], "openearthmap_eligibility.log")
    MANIFESTS["openearthmap"] = REPO_DIR / "data_local/oem_eligible7/samples.csv"
    setting.update(columns={}, domain_from_group_prefix=False)
    if not RUN_NAME.endswith("_eligible7"):
        RUN_NAME += "_eligible7"
    LOG_DIR = REPO_DIR / "notebook_logs" / RUN_NAME
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(REPO_DIR / "data_local/oem_eligible7/exclusions.json", LOG_DIR / "oem_exclusions.json")
    print("Run name:", RUN_NAME)
''')
md('''
## 5. Save resolved A100 configs and audit all data
BF16 AMP, frozen SAM feature caching, and batch size 2 are enabled. Tune batch/chunk sizes after Stage 0 if needed;
change `RUN_NAME` when changing settings. Resume requires an identical config.
The defaults retain the full 2,000-update/domain budget and 200-epoch ceiling. The epoch ceiling can stop a run before its update budget.
''')
code('''
CONFIGS, OUTPUTS = {}, {}
for dataset in DATASETS:
    setting = SETTINGS[dataset]
    output = REPO_DIR / "outputs" / RUN_NAME / dataset
    cfg_path = REPO_DIR / "configs" / "generated" / RUN_NAME / (dataset + ".yaml")
    overrides = {
        "output": str(output),
        "dataset": {"manifest": str(MANIFESTS[dataset]), "split_source": SPLIT_SOURCES[dataset],
                    **{k: setting[k] for k in ["columns", "default_domain", "domain_from_group_prefix",
                                               "label_map", "ignore_values", "color_map"]}},
        "runtime": {"device": "cuda", "amp": True, "batch_size": BATCH_SIZE,
                    "eval_batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS},
        "model": {"checkpoint": str(SAM), "adapter": "quantum", "quantum_device": "default.qubit",
                  "cache_dir": str(REPO_DIR / "cache" / "sam_vit_b" / dataset)},
        "training": {"epochs": EPOCHS, "max_updates_per_domain": MAX_UPDATES},
    }
    override_file = LOG_DIR / (dataset + "_overrides.json")
    override_file.write_text(json.dumps(overrides))
    run([PY, "-c", "import json,sys,yaml; from pathlib import Path; "
         "from qcl.config import load_config,merge; "
         "cfg=merge(load_config(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text())); "
         "p=Path(sys.argv[3]); p.parent.mkdir(parents=True,exist_ok=True); "
         "text=yaml.safe_dump(cfg,sort_keys=False); "
         "assert not p.exists() or p.read_text()==text, 'Config changed: choose a new RUN_NAME'; "
         "p.write_text(text); load_config(p); print(text)",
         REPO_DIR / "configs/datasets" / (dataset + ".yaml"), override_file, cfg_path], dataset + "_config.log")
    CONFIGS[dataset], OUTPUTS[dataset] = cfg_path, output
    run([PY, "audit_dataset.py", "--config", cfg_path,
         "--output", LOG_DIR / (dataset + "_audit")], dataset + "_audit.log")
''')
md('''
## 6. GPU profile and circuit check
Stage 0 loads the real SAM model but profiles adapter/decoder updates on synthetic features.
It excludes encoder, data loading, validation, and PCA time, so it is not an end-to-end runtime estimate.
''')
code('''
for dataset in DATASETS:
    args = [PY, "stage0.py", "--config", CONFIGS[dataset], "--repetitions", "5",
            "--output", LOG_DIR / (dataset + "_stage0.json")]
    if RUN_RANK_CHECK:
        args += ["--rank-check"]
    run(args, dataset + "_stage0.log")
''')
md('''
## 7. Train both experiments and evaluate their held-out splits
Training logs stream below and are saved to disk. Rerunning resumes `last.pt` at an epoch boundary.
Completed `final.pt` runs are reused. Interrupted runs without `last.pt` need a new `RUN_NAME`.
Evaluation retains the training split and checks data hashes. Keep the original dataset, SAM weights, and run folders in place.
''')
code('''
REPORTS = {}
for dataset in DATASETS:
    output = OUTPUTS[dataset]
    final, last = output / "final.pt", output / "last.pt"
    if not final.is_file():
        args = [PY, "train.py", "--config", CONFIGS[dataset]]
        if last.is_file():
            assert RESUME, "Existing run: enable RESUME or choose a fresh RUN_NAME"
            args += ["--resume", last]
        elif (output / "splits.json").exists():
            raise RuntimeError(f"Partial run without resumable checkpoint: choose a new RUN_NAME ({output})")
        run(args, dataset + "_train.log")
    else:
        print("Reusing completed training:", final)
    assert final.is_file(), f"Training did not produce {final}"
    # A new report directory avoids treating an interrupted evaluation as complete.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report = output / "test" / ("final_" + stamp)
    run([PY, "testing.py", "--checkpoint", final, "--output", report], dataset + "_test.log")
    REPORTS[dataset] = report
''')
md('''
## 8. Inspect results and download a compact report archive
The ZIP contains logs, configs and reports. Large checkpoints, feature caches, and datasets remain on the server.
Copy the entire run folder separately if you need checkpoints for later resume/evaluation.
''')
code('''
from IPython.display import display, Image, FileLink
import zipfile
for dataset in DATASETS:
    print("\\n===", dataset, "===")
    for path in [OUTPUTS[dataset] / "run_status.json", REPORTS[dataset] / "metrics.json"]:
        print(path.name)
        print(path.read_text()[:16000])
    for picture in list(REPORTS[dataset].rglob("*.png"))[:6]:
        display(Image(filename=str(picture), width=850))
archive = REPO_DIR / (RUN_NAME + "_reports.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    roots = [LOG_DIR, REPO_DIR / "configs/generated" / RUN_NAME, REPO_DIR / "outputs" / RUN_NAME]
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".png", ".svg", ".yaml", ".txt", ".log"}:
                z.write(path, path.relative_to(REPO_DIR))
print("Saved:", archive)
display(FileLink(archive.name))
print("If the link is outside Jupyter's served directory, download the ZIP with its file browser or SFTP.")
''')
md('''
## Troubleshooting
- **Missing columns / domain:** inspect the printed CSV header; set `columns` and verified domain metadata in Settings.
  If scene groups are absent, recover them from source metadata. Do not make each tile its own group.
- **Unknown labels:** check the dataset's preprocessing metadata and edit `label_map`, `ignore_values`, or `color_map`.
  An audit can detect unknown IDs but cannot establish what an ID means.
- **Group leakage / missing split partitions:** fix source metadata or define defensible merged domains before running.
- **CUDA out of memory:** choose a new run name, reduce `BATCH_SIZE` to 1 and, if needed, set `model.quantum_chunk` to 8 in overrides.
- **Slow first epoch:** SAM features are cached and PCA is initialized using training samples. Leave enough disk space for the cache.
- **Interrupted kernel:** rerun cells with the same settings. Resume checkpoints are written at epoch boundaries; unfinished epoch work is repeated.
- **Final status below 2,000 updates:** the epoch ceiling was reached. For a fresh full-budget experiment, raise `EPOCHS` with a new run name.

Prepared against the local project in `D:/QCL_Final_Boss/QCL_Final_Boss-main` and the
[repository](https://github.com/Arun2005-srm/QCL_Final_Boss). Notebook structure and Python syntax are checked locally;
GPU execution and your server's CSV contents must be validated on the server by the cells above.
''')

for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:02d}'
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3 (ipykernel)', language='python', name='python3'),
                language_info=dict(name='python', version='3.11.0')), nbformat=4, nbformat_minor=5)
path = Path(__file__).with_name('Run_Full_Project_A100.ipynb')
path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(path)
