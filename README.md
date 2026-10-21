# HCGL

The release keeps the main HERO method, DMSG, FTF-ER, MetaCLGraph, and the formal baselines that are supported by the common training entry point. Test files, ablation-only code, parameter sweeps, local results, checkpoints, figures, caches, editor settings, and development notes are not included.

## Environment

The original experiments used Python 3.8, PyTorch 2.3.0, and DGL 2.2.1. Create a clean environment and install the listed packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Install a PyTorch and DGL build that matches the CUDA version on your machine.

## Data

The datasets are not included because they are large and may have separate redistribution terms. Place the processed DBLP, IMDB, and Freebase folders under one data root. Each folder must follow the file layout described in [data/README.md](data/README.md). You can download data from https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing.

```text
DATA_ROOT/
├── DBLP/
├── IMDB/
└── Freebase/
```

No absolute local path is stored in the code. Set `DATA_ROOT` when running the reproduction script, or pass `--data-root` to `main.py`.

## Reproduce the main experiment

The following command runs HERO on all three datasets with five seeds and writes one JSON result per run, log files, checkpoints, and an aggregate CSV file:

```bash
DATA_ROOT=/path/to/processed/data \
DEVICE=cuda:0 \
bash scripts/reproduce_main.sh
```

The dataset task counts are DBLP=2, IMDB=5, and Freebase=7. DBLP uses class-balanced evaluation. MetaCLGraph uses patience 200. The default seeds are `0 1 2 3 4`. A smaller check can be run as follows:

```bash
DATA_ROOT=/path/to/processed/data \
DEVICE=cpu \
DATASETS="DBLP" \
SEEDS="0" \
bash scripts/reproduce_main.sh --num-epochs 2 --patience 2
```

Environment variables `METHOD`, `DATASETS`, `SEEDS`, `DEVICE`, `OUTPUT_DIR`, `CACHE_DIR`, and `PYTHON_BIN` can change a run without editing the script. Extra command-line arguments passed to the script are forwarded to `main.py`.

For one run:

```bash
python3 main.py \
  --dataset DBLP \
  --method hero \
  --num-tasks 2 \
  --seed 0 \
  --device cuda:0 \
  --data-root /path/to/processed/data \
  --output-dir outputs/main
```

## Output

Each JSON file records Average Performance (AP), Average Forgetting (AF), the full accuracy matrix, run times, the seed, and the main hyperparameters. `scripts/aggregate_results.py` computes the mean and sample standard deviation across seeds. Generated files are excluded by `.gitignore`.

## Release scope

We have released part of the core code, including the implementation using the HAN backbone. We will release the remaining code, including implementations for other backbones, once the paper is accepted.
