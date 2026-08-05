# Hướng dẫn H12: kiểm định paired cho H11

## Mục tiêu

H12 không huấn luyện lại mô hình. Thí nghiệm sử dụng 12 file prediction test đã
khóa của H11 để kiểm định bốn contrast:

1. Base-VI prosody so với Base-VI acoustic.
2. Large-VI prosody so với Large-VI acoustic.
3. Large-VI acoustic so với Base-VI acoustic.
4. Large-VI prosody so với Base-VI prosody.

Mỗi contrast chạy riêng cho seed 42, 43 và 44, trên cả nhiệm vụ region và
province. Script báo cáo accuracy, balanced accuracy, macro-F1, speaker-bootstrap
confidence interval 95%, xác suất candidate tốt hơn và exact McNemar. McNemar
được hiệu chỉnh Holm trong từng family 12 phép thử của mỗi nhiệm vụ.

## Điều kiện đầu vào

Từ thư mục gốc repository, kiểm tra đủ prediction:

```bash
for backbone in base large; do
  for variant in acoustic prosody; do
    for seed in 42 43 44; do
      file="outputs/h11_${backbone}_vi_${variant}_seed${seed}/predictions_test_best_province_accuracy.jsonl"
      test -s "$file" && echo "OK: $file" || echo "THIẾU: $file"
    done
  done
done
```

Phải có 12 dòng `OK`. H12 dừng nếu hai prediction trong một cặp không có cùng
filename, speaker ID hoặc ground truth.

## Chạy H12

H12 chỉ dùng CPU và không cần cấp GPU:

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
source /home/stackops/whale/envs/hierarchical-dialect-moe/bin/activate
mkdir -p logs results_archive/h12

nohup python -u scripts/analyze_h12.py \
  --outputs outputs \
  --destination results_archive/h12 \
  --seeds 42 43 44 \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 12026 \
  > logs/h12.log 2>&1 &

echo $! > logs/h12.pid
```

Theo dõi:

```bash
pid=$(cat logs/h12.pid)
ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd
tail -f logs/h12.log
```

## Artifact đầu ra

```text
results_archive/h12/h12_per_seed.csv
results_archive/h12/h12_aggregate.csv
results_archive/h12/h12_details.json
results_archive/h12/h12_summary.json
```

- `h12_per_seed.csv`: kết quả từng contrast, seed và task.
- `h12_aggregate.csv`: trung bình ba seed và số seed có CI không chứa 0.
- `h12_details.json`: toàn bộ bootstrap CI và bảng McNemar.
- `h12_summary.json`: protocol và manifest của phân tích.

## Kiểm tra sau khi chạy

```bash
test -s results_archive/h12/h12_per_seed.csv && echo OK_per_seed
test -s results_archive/h12/h12_aggregate.csv && echo OK_aggregate
test -s results_archive/h12/h12_details.json && echo OK_details
test -s results_archive/h12/h12_summary.json && echo OK_summary

wc -l results_archive/h12/h12_per_seed.csv
wc -l results_archive/h12/h12_aggregate.csv
```

Kỳ vọng `h12_per_seed.csv` có 25 dòng gồm header và 24 phép so sánh; bảng
aggregate có 9 dòng gồm header và 8 nhóm.

In phần province:

```bash
python - <<'PY'
import pandas as pd

path = "results_archive/h12/h12_aggregate.csv"
df = pd.read_csv(path)
cols = [
    "contrast",
    "runs",
    "baseline_accuracy_mean",
    "candidate_accuracy_mean",
    "difference_accuracy_mean",
    "difference_macro_f1_mean",
    "bootstrap_ci_excludes_zero_accuracy_runs",
    "mcnemar_holm_significant_runs",
]
print(df[df["task"] == "province"][cols].to_string(index=False))
PY
```

Không chọn mô hình hoặc điều chỉnh hyperparameter dựa trên H12. Đây là phân tích
xác nhận trên prediction test đã khóa.
