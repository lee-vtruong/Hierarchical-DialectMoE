# Hierarchical DialectMoE

Prosody-aware hierarchical Mixture-of-Experts for Vietnamese regional and
provincial dialect identification. The implementation targets the public
ViMD dataset (102.56 hours, 63 provinces).

The repository contains code and download scripts only. Audio, model weights,
checkpoints and experiment outputs are intentionally excluded from Git.

Các CSV/JSON thực nghiệm nhỏ đã phục hồi sau sự cố mất server được lưu tại
[`results_archive/`](results_archive/README.md), kèm checksum và ghi chú về những
artifact không thể phục hồi. Dataset, checkpoint và prediction đầy đủ vẫn không
được đưa vào Git.

## Model

The training path is:

```text
audio
  -> pretrained speech encoder
  -> masked temporal pooling
  -> acoustic projection --------------------+
                                                -> gated fusion
prosody (F0, energy, ZCR, spectrum) -> MLP ----+
  -> regional head (North/Central/South)
  -> region-conditioned sparse top-k router
  -> province experts
  -> 63-province head
```

The objective combines regional cross entropy, provincial cross entropy,
router entropy and expert load balancing. Evaluation reports accuracy,
balanced accuracy, per-class precision/recall/F1 and expert usage.

## Server installation

Python 3.10 or 3.11 is recommended. Install the CUDA build of PyTorch that
matches the server first, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the correct PyTorch CUDA wheel from pytorch.org first.
pip install -r requirements.txt
```

## Download data

ViMD is about 74.2 GB on Hugging Face. Allow at least 100 GB for the dataset
and generated cache:

```bash
python scripts/download_data.py --dataset vimd --output-dir data
```

Optional ViSpeech download:

```bash
python scripts/download_data.py --dataset vispeech --output-dir data
```

The ViMD downloader stores Parquet shards in `data/ViMD_Dataset`. Training
automatically uses this local copy. If the directory is absent, the Hugging
Face loader downloads/caches `nguyendv02/ViMD_Dataset` automatically.

ViMD is distributed under CC BY-NC-ND 4.0. Check that its non-commercial and
no-derivatives conditions fit the intended use before redistributing anything.
Do not commit the downloaded audio to GitHub.

## Train

Edit `configs/vimd_moe.yaml` for the server, then run:

```bash
python scripts/train.py --config configs/vimd_moe.yaml
```

Useful first server check after downloading the data:

```bash
python scripts/train.py --config configs/vimd_moe.yaml --max-samples 32
```

Checkpoints and label maps are written to `outputs/vimd_moe`. Resume with:

```bash
python scripts/train.py \
  --config configs/vimd_moe.yaml \
  --resume outputs/vimd_moe/last.pt
```

For a multi-GPU server, launch one process per GPU with `torchrun` after
adapting the training entry point to DDP, or run the current single-process
version on one GPU. The default configuration uses gradient accumulation and
mixed precision.

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/vimd_moe.yaml \
  --checkpoint outputs/vimd_moe/best.pt \
  --split test
```

Metrics are saved next to the checkpoint as `metrics_test.json`.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The unit tests cover prosody extraction, sparse expert routing, gradient flow
and the combined hierarchical loss.

## Recommended experiment order

1. Acoustic-only baseline (disable or zero prosody features).
2. Acoustic + prosody gated fusion without MoE.
3. Flat sparse MoE.
4. Region-conditioned hierarchical MoE.
5. Ablate F0, energy and spectral features.
6. Compare 2/4/8/16 experts and top-k 1/2.

Use at least three seeds and report both macro-F1 and balanced accuracy because
the 63 provincial classes are not perfectly balanced.

#
