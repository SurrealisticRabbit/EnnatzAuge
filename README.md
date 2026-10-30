# EnnatzAuge

Football video analysis: train YOLO models on GSR 2025, then run them over a
clip to get player positions in real pitch coordinates.

Two models, two datasets:

| dataset | task | what it learns |
| --- | --- | --- |
| `yolo-players` | detect | players, referees, goalkeepers, ball |
| `yolo-pitch` | pose | 29 pitch keypoints, used to solve the homography |

The pitch keypoints are the weak link — see [docs/data.md](docs/data.md) for the
measured state of that model before you trust any position output.

## Setup

```powershell
$env:PYTHONPATH = "C:\Users\ducky\Projects\EnnatzAuge\src"
python -m EnnatzAuge doctor --install
```

`doctor` is the first command to run on any new machine, and the only one you
need to run before training. It:

- **installs what is missing.** `--install` pip-installs the Python packages
  *and* torch. The torch wheel index is chosen from the detected backend
  (`rocm6.2` / `cu124` / `cpu`), because the default PyPI wheel is CPU-only on
  Windows and Linux — a GPU host that installs it silently trains on CPU.
- **creates missing directories and configs**, so a fresh clone is usable.
- **validates the hardware.** Reports the backend, every visible device with
  its VRAM, ROCm arch (`gfx942` etc.) or CUDA capability, multi-GPU topology
  (NVLink vs XGMI vs PCIe), free disk, and whether the datasets are built.
- **sets the process-wide wins** it can apply safely: TF32 and cuDNN autotuning
  on CUDA, and the ROCm environment (`MIOPEN_FIND_MODE=FAST`,
  `HIPBLASLT_ALLOW_TF32=1`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`,
  `HSA_ENABLE_SDMA=0`) on AMD.
- **recommends training settings** for the card it found — batch, workers,
  cache mode, and whether to tune.

Exit code is 0 when the machine can train and analyse, 1 when something is
blocking. Anything it will not fix automatically (a CPU-only torch on a GPU
host) is reported with the exact command instead.

```powershell
python -m EnnatzAuge doctor            # check only
python -m EnnatzAuge doctor --install  # check and install
python -m EnnatzAuge doctor --json     # machine-readable
python -m EnnatzAuge doctor --no-fix   # report only, change nothing
```

## Workflow

```
doctor  ->  build  ->  train  ->  status  ->  analyze  ->  exfil
```

```powershell
python -m EnnatzAuge paths                  # where everything lives
python -m EnnatzAuge doctor --install       # 1. make this machine ready
python -m EnnatzAuge build --stride 5       # 2. build datasets from SN-GSR-2025
python -m EnnatzAuge train                  # 3. tune, then train
python -m EnnatzAuge status                 # 4. what runs exist
python -m EnnatzAuge stop players           #    stop a run after this epoch
python -m EnnatzAuge analyze --video x.mp4  # 5. inference + render
python -m EnnatzAuge exfil                  # 6. package runs and video
```

`build` takes `--datasets players,pitch`, `--splits`, `--stride`,
`--pitch-images link|copy`, `--keep-archives`, `--raw-dir`.

`analyze` takes `--video`, `--players`, `--pitch`, `--output`, `--start`,
`--frames`, `--imgsz`, `--chunk`, `--scale`, `--smoothing`, `--no-pitch`,
`--no-track`, `--no-cut-detect`, `--compensate`.

## train

`train` reads `.EnnatzAuge/configs/train.yaml` and runs every job in it. Each
run does three things in order:

1. **Resolve the batch.** `batch: -1` profiles the card you actually
   allocated, so the same config is correct on a 48 GB L40 and a 192 GB
   MI300X. The profile is clamped to a ceiling (512 on ≥80 GB, 128 otherwise)
   and to the size of the training set.
2. **Tune.** Ultralytics' hyperparameter search runs for `tune_iterations`
   iterations of `tune_epochs` epochs each, then training continues with the
   winner. **On by default** (10 × 10) — the search is a rounding error next to
   an 80-epoch run, and it is the only way the first run on a new dataset is
   not a guess. Explicit `lr0`/`optimizer` in the config win over the search
   result. Set `tune_iterations: 0` to skip.
3. **Train**, with the resolved batch, workers, and tuned hyperparameters.

```powershell
python -m EnnatzAuge train --dry-run          # print the resolved plan, start nothing
python -m EnnatzAuge train --job players     # one job
python -m EnnatzAuge train --tune-iterations 0  # skip the search
```

Anything in the config can be overridden on the command line: `--epochs`,
`--imgsz`, `--batch`, `--gpus`, `--workers`, `--cache`, `--amp`, `--compile`,
`--multi-scale`, `--rect`, `--cos-lr`, `--deterministic`, `--optimizer`,
`--vram-gb`, `--vram-fraction`, `--tune-iterations`, `--tune-epochs`.

### Dataset size is taken into account

Yes. `dataset_size()` in `train/trainer.py` counts the training images the
resolved `data.yaml` points at — either the files in the `train` directory, or
the lines in a generated stride list — and `resolve_batch()` bounds the batch by
it on **both** the profiling and the explicit path:

```
batch 256/GPU bounded by dataset size (40); using 40
```

The reason is that a batch larger than the training set spends an epoch in a
single optimiser step, which distorts the LR schedule far more than the bigger
batch helps. A count of 0 (yaml missing, unreadable, or no `train` key) is
treated as "no bound" rather than as an empty dataset.

### Running several jobs at once

Jobs are admitted by VRAM budget, not by GPU count. Give a job `vram_gb` or
`vram_fraction` and it partitions memory instead of claiming a device, so on a
192 GB part you can list several jobs and they run side by side rather than
queueing:

```yaml
jobs:
  - name: players-m
    vram_gb: 48
    model: yolo26m.pt
  - name: players-l
    vram_gb: 48
    model: yolo26l.pt
```

```powershell
python -m EnnatzAuge train --vram-pool 192 --max-concurrent 4
```

`--vram-pool` overrides the detected pool size; `--max-concurrent` caps how
many jobs run at once. Jobs that declare no budget are treated as exclusive and
claim the whole pool.

## AMD / ROCm

The pipeline is tuned for AMD Instinct parts. On ROCm the entry point sets:

| variable | why |
| --- | --- |
| `MIOPEN_FIND_MODE=FAST` | skip exhaustive MIOpen kernel search |
| `HIPBLASLT_ALLOW_TF32=1` | TF32 matmuls, ~2x on fp32 paths |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | less fragmentation over a long run |
| `HSA_ENABLE_SDMA=0` | avoids the SDMA copy path that stalls on some parts |

`doctor` reports the ROCm arch, whether hipBLASLt/MIOpen are configured,
`rocm-smi` output, and whether FFmpeg has a `vaapi` device for hardware decode.

On a ≥80 GB card `doctor` recommends **concurrent jobs rather than one enormous
batch** — a 192 GB part is not a bigger L40, it is several L40s, and the win is
comparing validation curves side by side. It also suggests `cache='disk'`
(a 192 GB card still cannot hold a 4K frame set in RAM) and `compile=True`.

## Layout

```
.EnnatzAuge/          runtime root: configs, datasets, models, input_video
  configs/            train.yaml, analyze.yaml, experiment.yaml
  models/runs/        training runs
src/EnnatzAuge/
  cli/                command entry points
  train/              accelerator, jobs, manager, trainer
  core/               models, objects (pitch, tracking, homography), analyze
  render/             video overlays
docs/data.md          measured state of the pitch keypoint model
```

## Known limits

- **The pitch keypoints do not constrain a homography.** Every frame yields ~5
  RANSAC inliers with 1 redundant point, clustered in the centre circle. The
  fix is in the dataset or the training, not the analysis code. Read
  [docs/data.md](docs/data.md) before trusting positions.
- **Video decode is CPU-only.** `core/analyze.py` decodes with
  `cv2.VideoCapture` in a Python loop. `accelerator.hardware_decode_available()`
  and `accelerator.ffmpeg_hwaccel()` report what the host could do, but no
  hardware decode path is wired up yet.
- **`VirtualPitch.history` grows unboundedly** — one `FrameState` per frame
  with full object lists. Fine for a short clip, heavy on a 20k-frame one.
