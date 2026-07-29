# Hướng dẫn H6: audit leakage và speaker-disjoint split

## 1. Mục tiêu

H6 kiểm tra liệu cùng speaker, filename hoặc audio có xuất hiện ở nhiều split hay
không. Nếu có speaker overlap, mô hình có thể học nhận dạng người nói thay vì
khả năng tổng quát hóa phương ngữ.

Quy trình gồm:

1. Audit metadata read-only.
2. Audit thời lượng hoặc SHA-256 nếu cần.
3. Kiểm tra speaker có xung đột nhãn.
4. Tạo manifest speaker-disjoint.
5. Train acoustic-only và acoustic + prosody legacy trên split mới.

Không sao chép audio. Manifest chỉ chứa split gốc, row index và split mới.

## 2. Cập nhật server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
mkdir -p outputs/h6_split_audit data/splits logs
```

## 3. Audit metadata trước

Lệnh này không decode hoặc hash audio nên nhanh nhất:

```bash
python scripts/audit_dataset_splits.py \
  --config configs/vimd_moe.yaml \
  --output-dir outputs/h6_split_audit \
  --audio-mode none
```

Kết quả:

```text
outputs/h6_split_audit/audit_summary.json
outputs/h6_split_audit/overlap_details.json
outputs/h6_split_audit/speaker_label_conflicts.json
outputs/h6_split_audit/records.csv
```

Xem tóm tắt:

```bash
cat outputs/h6_split_audit/audit_summary.json
```

Kiểm tra số speaker xung đột nhãn:

```bash
python -c "
import json
p='outputs/h6_split_audit/speaker_label_conflicts.json'
d=json.load(open(p, encoding='utf-8'))
print('conflicting speakers:', len(d))
for speaker, labels in list(d.items())[:20]:
    print(speaker, labels)
"
```

Gửi `audit_summary.json` và `speaker_label_conflicts.json` trước khi build split.

## 4. Audit thời lượng

Nếu cần thống kê số giờ bị overlap, chạy lại với `duration`. Lệnh đọc header
audio nhưng không tính hash toàn bộ 60 GB:

```bash
python scripts/audit_dataset_splits.py \
  --config configs/vimd_moe.yaml \
  --output-dir outputs/h6_split_audit_duration \
  --audio-mode duration
```

## 5. Audit duplicate audio bằng SHA-256

Chỉ chạy sau metadata audit vì thao tác này phải đọc và hash toàn bộ audio, có thể
mất nhiều thời gian và tạo tải I/O lớn:

```bash
nohup .venv/bin/python -u scripts/audit_dataset_splits.py \
  --config configs/vimd_moe.yaml \
  --output-dir outputs/h6_split_audit_sha256 \
  --audio-mode sha256 \
  > logs/h6_audio_hash_audit.log 2>&1 &

echo $! > logs/h6_audio_hash_audit.pid
```

Theo dõi:

```bash
tail -f logs/h6_audio_hash_audit.log
```

## 6. Kết quả metadata audit

Audit split gốc ghi nhận:

```text
train: 15.023 utterance, 10.291 speaker
valid:  1.900 utterance,  1.320 speaker
test:   2.026 utterance,  1.344 speaker
```

Chỉ có hai speaker overlap, đều giữa valid và test:

```text
spk_73_0186: valid 1, test 2
spk_76_0219: valid 1, test 1
```

Không có speaker overlap với train, không filename trùng split và không speaker
xung đột nhãn. Vì leakage rất nhỏ, không rebuild ngẫu nhiên toàn bộ dataset. Dùng
chiến lược `preserve`: giữ nguyên split và chuyển ba utterance test của hai
speaker trên sang valid. Priority `train,valid,test` đảm bảo speaker đã thấy khi
train hoặc chọn mô hình không còn trong test.

## 7. Tạo speaker-disjoint manifest

Chỉ chạy khi `speaker_label_conflicts` bằng 0:

```bash
python scripts/build_speaker_disjoint_split.py \
  --records outputs/h6_split_audit/records.csv \
  --output data/splits/vimd_speaker_disjoint_seed42.csv \
  --summary outputs/h6_split_audit/speaker_disjoint_summary.json \
  --strategy preserve \
  --split-priority train,valid,test \
  --seed 42 \
  --train-ratio 0.793 \
  --valid-ratio 0.100 \
  --test-ratio 0.107
```

Nếu có xung đột nhãn, script chủ động dừng. Không dùng
`--allow-label-conflicts` trước khi xem từng trường hợp.

Kết quả mong đợi:

```text
moved_speakers: 2
moved_utterances: 3
train utterances: 15023
valid utterances: 1903
test utterances: 2023
```

## 8. Kiểm tra manifest

```bash
cat outputs/h6_split_audit/speaker_disjoint_summary.json
```

Kiểm tra speaker không trùng:

```bash
python -c "
import csv, collections
d=collections.defaultdict(set)
with open('data/splits/vimd_speaker_disjoint_seed42.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        d[r['speaker_id']].add(r['new_split'])
bad={s:v for s,v in d.items() if len(v)>1}
print('speakers:', len(d))
print('overlap after split:', len(bad))
assert not bad
"
```

Sinh báo cáo Markdown:

```bash
python scripts/summarize_split_audit.py \
  --audit outputs/h6_split_audit/audit_summary.json \
  --split-summary outputs/h6_split_audit/speaker_disjoint_summary.json \
  --output outputs/h6_split_audit/BAO_CAO_AUDIT.md
```

## 9. Lưu manifest lên GitHub

Manifest nhỏ và `.gitignore` cho phép track riêng `data/splits/*.csv` mà vẫn bỏ
qua toàn bộ audio:

```bash
git add data/splits/vimd_speaker_disjoint_seed42.csv
git commit -m "Add deterministic ViMD speaker-disjoint manifest"
git push origin main
```

Không commit `records.csv` vì file đó thuộc outputs và có thể chứa metadata chi
tiết không cần đưa vào repository.

## 10. Config H6 đã chuẩn bị

Acoustic-only:

```text
h6_speaker_disjoint_acoustic.yaml
h6_speaker_disjoint_acoustic_seed43.yaml
h6_speaker_disjoint_acoustic_seed44.yaml
```

Acoustic + prosody legacy, không MoE:

```text
h6_speaker_disjoint_prosody.yaml
h6_speaker_disjoint_prosody_seed43.yaml
h6_speaker_disjoint_prosody_seed44.yaml
```

Manifest đã được duyệt. Vì train split giữ nguyên, không cần train lại. Sáu config
được dùng để đánh giá các checkpoint cũ trên test đã sửa.

## 11. Đánh giá checkpoint cũ trên repaired test

Chạy tuần tự để không cần GPU trống cùng lúc:

```bash
CONFIGS=(
  h6_speaker_disjoint_acoustic
  h6_speaker_disjoint_acoustic_seed43
  h6_speaker_disjoint_acoustic_seed44
  h6_speaker_disjoint_prosody
  h6_speaker_disjoint_prosody_seed43
  h6_speaker_disjoint_prosody_seed44
)

CHECKPOINTS=(
  outputs/acoustic_only_seed42/best_province_accuracy.pt
  outputs/acoustic_only_seed43/best_province_accuracy.pt
  outputs/acoustic_only_seed44/best_province_accuracy.pt
  outputs/acoustic_prosody_seed42/best_province_accuracy.pt
  outputs/acoustic_prosody_seed43/best_province_accuracy.pt
  outputs/acoustic_prosody_seed44/best_province_accuracy.pt
)

nohup bash -c '
set -e
CONFIGS=(
  h6_speaker_disjoint_acoustic
  h6_speaker_disjoint_acoustic_seed43
  h6_speaker_disjoint_acoustic_seed44
  h6_speaker_disjoint_prosody
  h6_speaker_disjoint_prosody_seed43
  h6_speaker_disjoint_prosody_seed44
)
CHECKPOINTS=(
  outputs/acoustic_only_seed42/best_province_accuracy.pt
  outputs/acoustic_only_seed43/best_province_accuracy.pt
  outputs/acoustic_only_seed44/best_province_accuracy.pt
  outputs/acoustic_prosody_seed42/best_province_accuracy.pt
  outputs/acoustic_prosody_seed43/best_province_accuracy.pt
  outputs/acoustic_prosody_seed44/best_province_accuracy.pt
)
for i in "${!CONFIGS[@]}"; do
  .venv/bin/python -u scripts/run_experiment.py \
    --config "configs/experiments/${CONFIGS[$i]}.yaml" \
    --checkpoint "${CHECKPOINTS[$i]}" \
    --split test \
    --skip-train
done
' > logs/h6_repaired_test_evaluation.log 2>&1 &

echo $! > logs/h6_repaired_test_evaluation.pid
```

`evaluate.py` ghi artifacts vào output directory H6, không ghi đè metrics cũ
cạnh checkpoint nguồn.

Theo dõi:

```bash
tail -f logs/h6_repaired_test_evaluation.log
```

Tổng hợp acoustic:

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --pattern 'h6_speaker_disjoint_acoustic_seed*/metrics_test_best_province_accuracy.json' \
  --destination outputs/h6_acoustic_repaired_test.csv \
  --aggregate-destination outputs/h6_acoustic_repaired_test_aggregate.csv
```

Tổng hợp prosody:

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --pattern 'h6_speaker_disjoint_prosody_seed*/metrics_test_best_province_accuracy.json' \
  --destination outputs/h6_prosody_repaired_test.csv \
  --aggregate-destination outputs/h6_prosody_repaired_test_aggregate.csv
```

Paired H1 trên repaired test:

```bash
for seed in 42 43 44; do
  python scripts/compare_predictions.py \
    --baseline "outputs/h6_speaker_disjoint_acoustic_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --candidate "outputs/h6_speaker_disjoint_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --output "outputs/h6_prosody_vs_acoustic_repaired_test_seed${seed}.json" \
    --bootstrap-iterations 10000 \
    --seed "$seed"
done
```

## 12. Lưu ý phương pháp

- Split seed là 42 và phải giữ cố định.
- Mỗi speaker chỉ thuộc đúng một split.
- Chiến lược chính là minimal repair, không tái chia ngẫu nhiên toàn bộ dữ liệu.
- Chế độ `rebuild` stratify theo majority province chỉ dành cho nghiên cứu split
  mới hoàn toàn, không dùng trong H6 hiện tại.
- Nếu một speaker có nhiều province/region, phải điều tra trước.
- Loader từ chối manifest thiếu row, trùng row, sai index hoặc speaker xuất hiện
  ở nhiều split.
- Test mới không được dùng để chọn hyperparameter.

## 13. Kết quả cuối cùng

H6 đã hoàn tất trên repaired test gồm 2.023 utterance và 1.342 speaker:

| Mô hình | Province accuracy | Province balanced accuracy | Province macro-F1 |
|---|---:|---:|---:|
| Acoustic-only | 0,3946 ± 0,0148 | 0,3975 ± 0,0163 | 0,3911 ± 0,0131 |
| Acoustic + prosody | **0,4427 ± 0,0087** | **0,4469 ± 0,0090** | **0,4368 ± 0,0082** |
| Chênh lệch | **+0,0481** | **+0,0494** | **+0,0457** |

Kiểm định paired cho province accuracy có McNemar p lần lượt là 6,52e-7,
3,18e-4 và 1,16e-6 ở seed 42, 43 và 44. Cả ba đều qua ngưỡng Bonferroni
0,0167. Kết quả cấp vùng chưa có ý nghĩa thống kê.

Speaker-overlap repair chỉ làm thay đổi chênh lệch province accuracy khoảng
-0,009 điểm phần trăm so với test gốc. Do đó H1 được xác nhận và ảnh hưởng của
leakage đã phát hiện là không đáng kể.

Các artifact cần lưu:

- `h6_acoustic_repaired_test_aggregate.csv`
- `h6_prosody_repaired_test_aggregate.csv`
- `h6_prosody_vs_acoustic_repaired_test_seed42.json`
- `h6_prosody_vs_acoustic_repaired_test_seed43.json`
- `h6_prosody_vs_acoustic_repaired_test_seed44.json`
