# Hướng dẫn H8: phân tích theo thời lượng và độ tin cậy

## 1. Mục tiêu

H8 ghép sáu prediction H6 với audio metadata của repaired test để phân tích:

1. Prosody hiệu quả hơn ở audio ngắn hay dài.
2. Số mẫu được sửa đúng và bị làm sai trong từng bucket thời lượng.
3. Accuracy và calibration gap trong từng mức confidence.
4. Hành vi của các tỉnh cải thiện mạnh: 17, 30, 22.
5. Hành vi của các tỉnh suy giảm nhất quán: 38, 70, 14, 11.

Không chạy lại mô hình và không cần GPU. Script chỉ đọc header của 2.023 audio để
lấy thời lượng, sau đó xử lý prediction bằng CPU.

## 2. Cập nhật server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
mkdir -p outputs/h8 logs
```

## 3. Chạy H8

Chạy trực tiếp:

```bash
python scripts/analyze_h8.py \
  --config configs/experiments/h6_speaker_disjoint_acoustic.yaml \
  --split test \
  --baseline-template 'outputs/h6_speaker_disjoint_acoustic_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --candidate-template 'outputs/h6_speaker_disjoint_prosody_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --seeds 42 43 44 \
  --duration-edges 0 2 4 6 10 20 \
  --confidence-edges 0 0.4 0.6 0.8 1 \
  --focus-provinces 17 30 22 38 70 14 11 \
  --output-dir outputs/h8
```

Quá trình đọc audio có thể mất vài phút. Có thể dùng `nohup`:

```bash
nohup .venv/bin/python -u scripts/analyze_h8.py \
  --config configs/experiments/h6_speaker_disjoint_acoustic.yaml \
  --split test \
  --baseline-template 'outputs/h6_speaker_disjoint_acoustic_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --candidate-template 'outputs/h6_speaker_disjoint_prosody_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --seeds 42 43 44 \
  --duration-edges 0 2 4 6 10 20 \
  --confidence-edges 0 0.4 0.6 0.8 1 \
  --focus-provinces 17 30 22 38 70 14 11 \
  --output-dir outputs/h8 \
  > logs/h8_analysis.log 2>&1 &
echo $! > logs/h8_analysis.pid
```

Theo dõi:

```bash
tail -f logs/h8_analysis.log
```

Kiểm tra tiến trình:

```bash
pid=$(cat logs/h8_analysis.pid)
ps -p "$pid" -o pid=,etime=,%cpu=,%mem=,stat=,cmd=
```

## 4. Kiểm tra kết quả

```bash
ls -lh outputs/h8
python -m json.tool outputs/h8/h8_summary.json | less
cat outputs/h8/duration_bucket_aggregate.csv
cat outputs/h8/confidence_bucket_aggregate.csv
cat outputs/h8/focus_province_aggregate.csv
```

Artifact sinh ra:

```text
outputs/h8/
├── h8_summary.json
├── duration_metadata.csv
├── duration_bucket_per_seed.csv
├── duration_bucket_aggregate.csv
├── confidence_bucket_per_seed.csv
├── confidence_bucket_aggregate.csv
├── focus_province_per_seed.csv
└── focus_province_aggregate.csv
```

`raw_duration_seconds` là thời lượng thật. `effective_duration_seconds` là thời
lượng mô hình thực sự thấy sau khi giới hạn theo `data.max_seconds = 20`. Phân
tích bucket dùng effective duration để phù hợp với input của mô hình.

Các mẫu có effective duration đúng 20 giây được ghi là `20 (capped)`, tách khỏi
bucket `[10,20)`. Phần lớn nhóm capped là audio gốc dài hơn 20 giây và đã bị cắt.

## 5. Cách đọc kết quả

- `improvement_mean > 0`: prosody tốt hơn acoustic trong bucket.
- `fixed_mean > regressed_mean`: prosody có lợi ròng.
- `calibration_gap > 0`: mô hình quá tự tin trong bucket.
- `calibration_gap < 0`: mô hình thiếu tự tin.
- Chỉ kết luận xu hướng khi có đủ support và lặp lại qua ba seed.

Không dùng H8 để chọn lại bucket hoặc hyperparameter dựa trên test. Các bucket đã
được khai báo trước theo khoảng thời lượng dễ diễn giải.

## 6. File cần gửi lại

Sau khi hoàn tất, tải và gửi:

```text
outputs/h8/h8_summary.json
outputs/h8/duration_bucket_aggregate.csv
outputs/h8/confidence_bucket_aggregate.csv
outputs/h8/focus_province_aggregate.csv
```

Không cần gửi `duration_metadata.csv` vì chứa 2.023 dòng và không cần thiết cho
kết luận tổng hợp.

## 7. Kết quả H8 đã thu được

- Thời lượng gốc trung bình 19,32 giây, trung vị 19,52 giây.
- 974/2.023 mẫu (48,15%) dài hơn 20 giây và bị cắt trước khi vào mô hình.
- Prosody cải thiện khoảng 3,97--5,12 điểm phần trăm ở các bucket có đủ support
  từ 6 giây trở lên.
- Hai bucket 2--4 và 4--6 giây chỉ có 13 và 18 mẫu, không đủ để kết luận.
- Cả hai mô hình đều quá tự tin ở mọi confidence bucket.
- Prosody giảm calibration gap ở cả bốn confidence bucket.
- Thời lượng trung bình của nhóm tỉnh cải thiện và suy giảm khá gần nhau; thời
  lượng không giải thích trực tiếp sự khác biệt theo tỉnh.
