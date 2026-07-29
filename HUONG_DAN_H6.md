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

## 6. Tạo speaker-disjoint manifest

Chỉ chạy khi `speaker_label_conflicts` bằng 0:

```bash
python scripts/build_speaker_disjoint_split.py \
  --records outputs/h6_split_audit/records.csv \
  --output data/splits/vimd_speaker_disjoint_seed42.csv \
  --summary outputs/h6_split_audit/speaker_disjoint_summary.json \
  --seed 42 \
  --train-ratio 0.793 \
  --valid-ratio 0.100 \
  --test-ratio 0.107
```

Nếu có xung đột nhãn, script chủ động dừng. Không dùng
`--allow-label-conflicts` trước khi xem từng trường hợp.

## 7. Kiểm tra manifest

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

## 8. Lưu manifest lên GitHub

Manifest nhỏ và `.gitignore` cho phép track riêng `data/splits/*.csv` mà vẫn bỏ
qua toàn bộ audio:

```bash
git add data/splits/vimd_speaker_disjoint_seed42.csv
git commit -m "Add deterministic ViMD speaker-disjoint manifest"
git push origin main
```

Không commit `records.csv` vì file đó thuộc outputs và có thể chứa metadata chi
tiết không cần đưa vào repository.

## 9. Config H6 đã chuẩn bị

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

Chưa chạy sáu config cho đến khi audit và phân bố split mới được duyệt.

## 10. Lưu ý phương pháp

- Split seed là 42 và phải giữ cố định.
- Mỗi speaker chỉ thuộc đúng một split.
- Stratification dùng majority province của speaker.
- Nếu một speaker có nhiều province/region, phải điều tra trước.
- Loader từ chối manifest thiếu row, trùng row, sai index hoặc speaker xuất hiện
  ở nhiều split.
- Test mới không được dùng để chọn hyperparameter.

