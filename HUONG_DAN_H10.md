# Hướng dẫn H10: multi-crop inference cho audio dài

## 1. Mục tiêu

H8 phát hiện 974/2.023 repaired-test audio dài hơn 20 giây và bị cắt. H10 giữ
nguyên checkpoint acoustic + prosody, chỉ thay chiến lược inference:

- `first`: 20 giây đầu, tương đương pipeline hiện tại.
- `start_end`: lấy đoạn 20 giây đầu và 20 giây cuối.
- `uniform`: lấy ba crop phân bố đều từ đầu đến cuối.

Logit của các crop được lấy trung bình trước softmax. Audio không dài hơn 20 giây
chỉ có một crop và không bị thay đổi.

H10 không huấn luyện lại và không dùng kết quả test để chỉnh model. Chiến lược
được so sánh paired với prediction H6.

## 2. Cập nhật server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
mkdir -p outputs/h10 logs
```

## 3. Chọn ba GPU trống

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

Ví dụ bên dưới dùng GPU 1, 2, 6. Thay bằng GPU đang trống.

## 4. Chạy control `first`, `start_end` và `uniform` cho ba seed

`first` là sanity control của evaluator mới. Prediction `first` phải có argmax
giống H6 trước khi diễn giải hai chiến lược multi-crop.

Seed 42:

```bash
nohup env CUDA_VISIBLE_DEVICES=1 bash -c '
set -e
for strategy in first start_end uniform; do
  .venv/bin/python -u scripts/evaluate_multicrop.py \
    --config configs/experiments/h6_speaker_disjoint_prosody.yaml \
    --checkpoint outputs/acoustic_prosody_seed42/best_province_accuracy.pt \
    --split test \
    --strategy "$strategy" \
    --uniform-crops 3 \
    --output-dir outputs/h10/prosody_seed42
done
' > logs/h10_seed42.log 2>&1 &
echo $! > logs/h10_seed42.pid
```

Seed 43:

```bash
nohup env CUDA_VISIBLE_DEVICES=2 bash -c '
set -e
for strategy in first start_end uniform; do
  .venv/bin/python -u scripts/evaluate_multicrop.py \
    --config configs/experiments/h6_speaker_disjoint_prosody_seed43.yaml \
    --checkpoint outputs/acoustic_prosody_seed43/best_province_accuracy.pt \
    --split test \
    --strategy "$strategy" \
    --uniform-crops 3 \
    --output-dir outputs/h10/prosody_seed43
done
' > logs/h10_seed43.log 2>&1 &
echo $! > logs/h10_seed43.pid
```

Seed 44:

```bash
nohup env CUDA_VISIBLE_DEVICES=6 bash -c '
set -e
for strategy in first start_end uniform; do
  .venv/bin/python -u scripts/evaluate_multicrop.py \
    --config configs/experiments/h6_speaker_disjoint_prosody_seed44.yaml \
    --checkpoint outputs/acoustic_prosody_seed44/best_province_accuracy.pt \
    --split test \
    --strategy "$strategy" \
    --uniform-crops 3 \
    --output-dir outputs/h10/prosody_seed44
done
' > logs/h10_seed44.log 2>&1 &
echo $! > logs/h10_seed44.pid
```

Nếu OOM, thêm `--batch-size 2`. Do mỗi sample có tối đa ba crop, memory cao hơn
inference H6.

## 5. Theo dõi

```bash
for seed in 42 43 44; do
  pid=$(cat "logs/h10_seed${seed}.pid")
  if ps -p "$pid" > /dev/null; then
    echo "seed ${seed}: ĐANG CHẠY PID=${pid}"
  else
    echo "seed ${seed}: ĐÃ DỪNG"
  fi
  tail -n 4 "logs/h10_seed${seed}.log"
  echo
done
```

Kiểm tra lỗi:

```bash
grep -HniE \
  "out of memory|CUDA out of memory|Traceback|RuntimeError|Error|Killed" \
  logs/h10_seed*.log
```

## 6. Xác nhận đủ artifact

```bash
for seed in 42 43 44; do
  for strategy in first start_end uniform; do
    file="outputs/h10/prosody_seed${seed}/predictions_test_${strategy}.jsonl"
    test -f "$file" && echo "OK $file" || echo "THIẾU $file"
  done
done
```

## 7. Xác nhận sanity control

```bash
python -c "
import json
from pathlib import Path
for seed in (42,43,44):
    old=Path(f'outputs/h6_speaker_disjoint_prosody_seed{seed}/predictions_test_best_province_accuracy.jsonl')
    new=Path(f'outputs/h10/prosody_seed{seed}/predictions_test_first.jsonl')
    a={r['filename']:r for r in map(json.loads,old.open(encoding='utf-8'))}
    b={r['filename']:r for r in map(json.loads,new.open(encoding='utf-8'))}
    assert set(a)==set(b), f'seed {seed}: sample mismatch'
    changed=sum(a[k]['province_pred_id'] != b[k]['province_pred_id'] for k in a)
    print(f'seed {seed}: changed province predictions = {changed}')
    assert changed==0, f'seed {seed}: first control does not reproduce H6'
"
```

Cả ba seed phải báo `changed province predictions = 0`. Nếu khác 0, dừng và gửi
log; không chạy kiểm định multi-crop.

## 8. So sánh paired với first-crop control

```bash
for seed in 42 43 44; do
  for strategy in start_end uniform; do
    python scripts/compare_predictions.py \
      --baseline "outputs/h10/prosody_seed${seed}/predictions_test_first.jsonl" \
      --candidate "outputs/h10/prosody_seed${seed}/predictions_test_${strategy}.jsonl" \
      --output "outputs/h10/${strategy}_vs_first_seed${seed}.json" \
      --bootstrap-iterations 10000 \
      --seed "$seed"
  done
done
```

## 9. Tổng hợp metrics

```bash
python scripts/summarize_experiments.py \
  --outputs outputs/h10 \
  --pattern 'prosody_seed*/metrics_test_*.json' \
  --destination outputs/h10/h10_summary.csv \
  --aggregate-destination outputs/h10/h10_aggregate.csv
```

Nếu script tổng hợp cũ không nhận metrics H10, dùng:

```bash
python -c "
import csv, json
from pathlib import Path
rows=[]
for p in sorted(Path('outputs/h10').glob('prosody_seed*/metrics_test_*.json')):
    d=json.load(open(p, encoding='utf-8'))
    rows.append({
        'seed': p.parent.name.split('seed')[-1],
        'strategy': d['strategy'],
        'samples': d['samples'],
        'mean_crops_per_sample': d['mean_crops_per_sample'],
        'province_accuracy': d['province']['accuracy'],
        'province_balanced_accuracy': d['province']['balanced_accuracy'],
        'province_macro_f1': d['province']['macro_f1'],
    })
with open('outputs/h10/h10_summary.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
print(*rows, sep='\n')
"
```

## 10. File cần gửi lại

```text
outputs/h10/h10_summary.csv
outputs/h10/prosody_seed42/metrics_test_first.json
outputs/h10/prosody_seed43/metrics_test_first.json
outputs/h10/prosody_seed44/metrics_test_first.json
outputs/h10/start_end_vs_first_seed42.json
outputs/h10/start_end_vs_first_seed43.json
outputs/h10/start_end_vs_first_seed44.json
outputs/h10/uniform_vs_first_seed42.json
outputs/h10/uniform_vs_first_seed43.json
outputs/h10/uniform_vs_first_seed44.json
```

Không chọn chiến lược chỉ vì một seed tốt hơn. Chiến lược chỉ được xem là cải
thiện nếu chênh lệch lặp lại qua seed và kiểm định paired ủng hộ.

Temperature H9 được fit cho first-crop logits, không áp dụng trực tiếp cho
multi-crop logits. H10 trước tiên chọn kết luận về accuracy/balanced
accuracy/macro-F1. Chỉ khi multi-crop thắng ổn định mới sinh validation
multi-crop prediction và fit lại temperature riêng.

## 11. Kết quả H10 đã thu được

| Chiến lược | Crop TB | Province accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| First | 1,000 | 0,4427 | 0,4469 | 0,4368 |
| Start-end | 1,481 | 0,4452 | 0,4490 | 0,4395 |
| Uniform-3 | 1,963 | 0,4442 | 0,4481 | 0,4388 |

Start-end tăng trung bình khoảng 0,25 điểm phần trăm accuracy nhưng không lặp lại
theo seed. Seed 42 gần như không đổi, seed 43 giảm nhẹ và seed 44 tăng. McNemar
seed 44 có p = 0,0479 nhưng không qua Bonferroni 0,0083 cho sáu so sánh.

Không chiến lược multi-crop nào được chấp nhận thay first-crop. Pipeline chính
giữ first-crop; không fit lại temperature cho multi-crop. Uniform-3 không phù hợp
vì gần gấp đôi compute nhưng kém start-end về trung bình.
