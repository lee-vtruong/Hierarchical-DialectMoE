# H16: Controlled ViP-VL / ChunkFormer baseline

## 1. Mục tiêu

H16 fine-tune encoder chính thức `khanhld/vip-vl-base-vie` trên đúng
speaker-disjoint repaired split của dự án. Encoder có 12 ChunkFormer blocks,
hidden size 512, 8 heads, 8x subsampling và nhận 80-bin log-Mel. Hai head mới
dự đoán 3 vùng và 63 tỉnh. Đây là controlled screening; upstream hiện hard-code
seed 777. Chỉ mở multi-seed nếu screening cạnh tranh với H11.

## 2. Pull và kiểm tra code

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
git pull --ff-only origin main
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe

python -m pytest -q tests/test_h16_conversion.py tests/test_components.py
```

## 3. Lấy implementation chính thức và ghi provenance

```bash
mkdir -p external
git clone https://github.com/khanld/chunkformer.git external/chunkformer
git -C external/chunkformer rev-parse HEAD | tee results_archive/h16_chunkformer_commit.txt

export CHUNKFORMER_DIR="$PWD/external/chunkformer"
export PYTHONPATH="$CHUNKFORMER_DIR:${PYTHONPATH:-}"
python -c "import chunkformer; print('ChunkFormer import OK')"
```

Không commit `external/chunkformer` vào repository chính.

Kiểm tra dependency trước; chỉ cài gói thực sự thiếu:

```bash
python - <<'PY'
mods = ['torch', 'torchaudio', 'yaml', 'pandas', 'tqdm', 'huggingface_hub', 'tensorboard', 'jiwer', 'librosa']
for name in mods:
    try:
        module = __import__(name)
        print('OK', name, getattr(module, '__version__', ''))
    except Exception as exc:
        print('MISSING', name, repr(exc))
PY
```

Nếu có `MISSING`, cài đúng các package đó; không chạy `pip install -e` vì bản
upstream khai báo Python >=3.11 trong khi environment dự án có thể là 3.10.

## 4. Download checkpoint ViP-VL

```bash
mkdir -p external/vip-vl-base-vie

huggingface-cli download khanhld/vip-vl-base-vie \
  --local-dir external/vip-vl-base-vie

ls -lh external/vip-vl-base-vie/{pytorch_model.pt,config.yaml,global_cmvn}
```

Checkpoint chứa pickle. Chỉ tải từ repository chính thức và dùng PyTorch >=2.6.

## 5. Export đúng repaired split

Lệnh này tạo FLAC riêng cho recipe ChunkFormer. Cần kiểm tra dung lượng trước:

```bash
df -h /

python -u scripts/prepare_h16_chunkformer.py \
  --config configs/experiments/h11_large_vi_prosody.yaml \
  --destination data/h16_chunkformer \
  2>&1 | tee logs/h16_prepare.log
```

Chuyển TSV sang format JSONL của upstream và dùng global CMVN từ pretraining:

```bash
for split in train dev test; do
  python external/chunkformer/tools/tsv_to_list.py \
    "data/h16_chunkformer/${split}/data.tsv"
done

cp external/vip-vl-base-vie/global_cmvn \
  data/h16_chunkformer/train/global_cmvn
```

Xác nhận số mẫu:

```bash
wc -l data/h16_chunkformer/{train,dev,test}/data.list
wc -l data/h16_chunkformer/metadata.jsonl
du -sh data/h16_chunkformer
```

Kỳ vọng repaired test có 2.023 dòng. `data.tsv` có thêm một header nhưng
`data.list` không có header.

## 6. Smoke test official recipe

Tạo subset mà không sửa bộ dữ liệu đầy đủ:

```bash
mkdir -p data/h16_smoke/{train,dev}
head -n 16 data/h16_chunkformer/train/data.list > data/h16_smoke/train/data.list
head -n 16 data/h16_chunkformer/dev/data.list > data/h16_smoke/dev/data.list

export CHUNKFORMER_DIR="$PWD/external/chunkformer"
export PYTHONPATH="$CHUNKFORMER_DIR:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone --nnodes=1 --nproc-per-node=1 \
  external/chunkformer/chunkformer/bin/train.py \
  --use_amp \
  --train_engine torch_ddp \
  --config configs/experiments/h16_vipvl_multitask.yaml \
  --override_config "max_epoch 1" \
  --data_type raw \
  --train_data data/h16_smoke/train/data.list \
  --cv_data data/h16_smoke/dev/data.list \
  --checkpoint external/vip-vl-base-vie/pytorch_model.pt \
  --model_dir outputs/h16_vipvl_smoke \
  --tensorboard_dir outputs/h16_vipvl_smoke/tensorboard \
  --num_workers 0
```

Nếu CLI báo cách viết `--override_config`, chạy
`python external/chunkformer/chunkformer/bin/train.py -h` và gửi output; không
chạy full trước khi smoke tạo checkpoint.

## 7. Full training seed 777

```bash
mkdir -p logs outputs/h16_vipvl_seed777

nohup env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD/external/chunkformer:${PYTHONPATH:-}" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  external/chunkformer/chunkformer/bin/train.py \
  --use_amp \
  --train_engine torch_ddp \
  --config configs/experiments/h16_vipvl_multitask.yaml \
  --data_type raw \
  --train_data data/h16_chunkformer/train/data.list \
  --cv_data data/h16_chunkformer/dev/data.list \
  --checkpoint external/vip-vl-base-vie/pytorch_model.pt \
  --model_dir outputs/h16_vipvl_seed777 \
  --tensorboard_dir outputs/h16_vipvl_seed777/tensorboard \
  --num_workers 8 \
  --pin_memory \
  > logs/h16_vipvl_seed777.log 2>&1 &

echo $! | tee logs/h16_vipvl_seed777.pid
```

Theo dõi:

```bash
tail -f logs/h16_vipvl_seed777.log
watch -n 5 nvidia-smi
```

## 8. Average checkpoint và đánh giá test

Sau khi đủ 30 epoch:

```bash
export CHUNKFORMER_DIR="$PWD/external/chunkformer"
export PYTHONPATH="$CHUNKFORMER_DIR:${PYTHONPATH:-}"

python external/chunkformer/chunkformer/bin/average_model.py \
  --dst_model outputs/h16_vipvl_seed777/avg_10.pt \
  --src_path outputs/h16_vipvl_seed777 \
  --num 10

python external/chunkformer/chunkformer/bin/classify.py \
  --gpu 0 \
  --config outputs/h16_vipvl_seed777/train.yaml \
  --data_type raw \
  --test_data data/h16_chunkformer/test/data.list \
  --checkpoint outputs/h16_vipvl_seed777/avg_10.pt \
  --batch_size 16 \
  --result_dir outputs/h16_vipvl_seed777/test \
  --dtype fp16
```

Chuyển prediction về schema của dự án:

```bash
python scripts/convert_h16_predictions.py \
  --predictions outputs/h16_vipvl_seed777/test/predictions.tsv \
  --metadata data/h16_chunkformer/metadata.jsonl \
  --split test \
  --output outputs/h16_vipvl_seed777/predictions_test.jsonl
```

## 9. Paired comparison với H11

Vì upstream dùng seed 777, screening so với cả ba H11 seed để tránh chọn một
baseline initialization thuận lợi:

```bash
mkdir -p results_archive/h16
for seed in 42 43 44; do
  python scripts/compare_predictions.py \
    --baseline "outputs/h11_large_vi_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --candidate outputs/h16_vipvl_seed777/predictions_test.jsonl \
    --output "results_archive/h16/h16_vs_h11_seed${seed}.json" \
    --bootstrap-iterations 10000 \
    --seed 16000
done
```

## 10. Gate quyết định

- Nếu province macro-F1 dưới 57,17%, trước tiên kiểm tra protocol/config vì thấp
  hơn published ViP-VL.
- Nếu đạt 57,17%--59,0%, implementation hợp lý nhưng chưa vượt H11; giữ làm
  controlled strong baseline.
- Nếu xấp xỉ hoặc vượt 59,5%, patch upstream để nhận seed runtime và chạy ít
  nhất ba seed trước khi kết luận.
- Không thay đổi test/crop/config sau khi nhìn test. Mọi sửa đổi tiếp theo phải
  được chọn trên dev.
