# Hướng dẫn H7: phân tích lỗi và calibration

## 1. Mục tiêu

H7 không huấn luyện lại mô hình và không cần GPU. Script dùng prediction của H6
trên repaired speaker-disjoint test để trả lời:

1. Tỉnh nào được cải thiện hoặc suy giảm khi thêm prosody?
2. Prosody sửa đúng bao nhiêu mẫu acoustic-only dự đoán sai và làm hỏng bao nhiêu
   mẫu vốn dự đoán đúng?
3. Các cặp tỉnh nào thường bị nhầm nhất?
4. Mô hình có quá tự tin hay không qua ECE, NLL và Brier score?
5. Kết quả có ổn định qua seed 42, 43 và 44 không?

## 2. Cập nhật và kiểm tra

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
```

Kiểm tra đủ sáu prediction:

```bash
for seed in 42 43 44; do
  test -f "outputs/h6_speaker_disjoint_acoustic_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    && echo "OK acoustic seed ${seed}" \
    || echo "THIẾU acoustic seed ${seed}"
  test -f "outputs/h6_speaker_disjoint_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    && echo "OK prosody seed ${seed}" \
    || echo "THIẾU prosody seed ${seed}"
done
```

Nếu cả sáu dòng đều báo `OK`, chạy bước tiếp theo.

## 3. Chạy H7

Lệnh chạy trực tiếp:

```bash
python scripts/analyze_h7.py \
  --baseline-template 'outputs/h6_speaker_disjoint_acoustic_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --candidate-template 'outputs/h6_speaker_disjoint_prosody_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --seeds 42 43 44 \
  --calibration-bins 10 \
  --output-dir outputs/h7
```

Lệnh này chạy bằng CPU và thường không cần `nohup`. Nếu vẫn muốn chạy nền:

```bash
mkdir -p logs
nohup .venv/bin/python -u scripts/analyze_h7.py \
  --baseline-template 'outputs/h6_speaker_disjoint_acoustic_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --candidate-template 'outputs/h6_speaker_disjoint_prosody_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --seeds 42 43 44 \
  --calibration-bins 10 \
  --output-dir outputs/h7 \
  > logs/h7_analysis.log 2>&1 &
echo $! > logs/h7_analysis.pid
```

Theo dõi:

```bash
tail -f logs/h7_analysis.log
```

## 4. Kiểm tra kết quả

```bash
ls -lh outputs/h7
python -m json.tool outputs/h7/h7_summary.json | less
head -n 11 outputs/h7/province_aggregate.csv
head -n 21 outputs/h7/confusion_pairs.csv
cat outputs/h7/calibration_per_seed.csv
```

Các artifact:

| File | Nội dung |
|---|---|
| `h7_summary.json` | Kết luận tóm tắt, top tỉnh và top cặp nhầm |
| `province_per_seed.csv` | Kết quả từng tỉnh ở từng seed |
| `province_aggregate.csv` | Trung bình và độ lệch chuẩn qua ba seed |
| `confusion_pairs.csv` | Số lỗi theo cặp tỉnh trước/sau khi thêm prosody |
| `calibration_per_seed.csv` | ECE, NLL, Brier và confidence từng seed |
| `calibration_bins.csv` | Chi tiết từng confidence bin |

Trong `province_aggregate.csv`:

- `improvement_mean > 0`: prosody cải thiện tỉnh đó.
- `improved_seeds = 3`: cải thiện nhất quán ở cả ba seed.
- `degraded_seeds = 3`: suy giảm nhất quán, cần điều tra.
- `fixed_total`: số dự đoán sai của acoustic được prosody sửa đúng.
- `regressed_total`: số dự đoán đúng của acoustic bị prosody làm sai.

Với calibration, giá trị ECE, NLL và Brier càng thấp càng tốt. Không kết luận mô
hình calibration tốt hơn chỉ từ confidence trung bình.

## 5. Gửi kết quả để bổ sung báo cáo

Tải về và gửi các file:

```text
outputs/h7/h7_summary.json
outputs/h7/province_aggregate.csv
outputs/h7/confusion_pairs.csv
outputs/h7/calibration_per_seed.csv
```

Không đưa toàn bộ prediction hoặc checkpoint lên GitHub vì chúng lớn và đã có
trên server.

## 6. Giới hạn

Prediction H6 hiện không chứa thời lượng audio. Vì vậy H7 này chưa phân tích theo
duration để tránh suy diễn dữ liệu không tồn tại. Nếu kết quả H7 chính đã ổn định,
có thể bổ sung một script chỉ đọc metadata/header audio và ghép thời lượng theo
`filename` mà không cần chạy lại mô hình.

## 7. Kết quả H7 đã thu được

- Prosody sửa đúng 841 lượt và làm sai 549 lượt qua ba seed, lợi ròng 292.
- Accuracy tăng ở 34/63 tỉnh, giảm ở 23 và không đổi ở 6.
- 17 tỉnh cải thiện ở cả ba seed; 6 tỉnh suy giảm ở cả ba seed.
- Tỉnh cải thiện lớn nhất: 17, 30, 22, 81 và 77.
- Tỉnh suy giảm nhất quán lớn nhất: 38, 70, 14 và 11.
- ECE trung bình giảm từ 0,2918 xuống 0,2453.
- NLL trung bình giảm từ 2,8241 xuống 2,4748.
- Brier trung bình giảm từ 0,8569 xuống 0,7820.
- Calibration tốt hơn ở seed 42 và 44, nhưng ECE/NLL xấu hơn ở seed 43.

Không diễn giải các chênh lệch từng tỉnh như kiểm định độc lập vì mỗi tỉnh chỉ có
khoảng vài chục mẫu. Kết luận đáng tin cậy nhất vẫn là kết quả paired toàn bộ test
ở H6; H7 dùng để định vị nhóm lỗi và tạo giả thuyết cho thí nghiệm tiếp theo.
