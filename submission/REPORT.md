# Lab 21 — Evaluation Report

**Họ tên**: Đào Quốc Đại  
**MSSV**: 2A202601285  
**Ngày**: 21/08/2026  
**Tier**: `T4`  
**Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: `Tesla T4, 14.6 GB VRAM`

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket chăm sóc khách hàng → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được = 98, suggested = 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** Có.

`template_check.json` cho kết quả `ok=true`, đồng thời giữ được cả `<think>`, nội dung reasoning và `</think>`. Vì vậy chat template không làm mất reasoning span.

Về `max_length`, thống kê token cho thấy mean=93.1, p95=98, p99=100 và max=101, nên 256 đã đủ cho corpus hiện tại. Tôi vẫn giữ `max_length=1024` theo cấu hình chuẩn của tier T4 để giữ cấu hình thực nghiệm nhất quán giữa các notebook. Việc dùng 1024 lớn hơn nhu cầu đo được nhưng không làm thay đổi tính công bằng giữa các run vì tất cả đều dùng cùng cấu hình.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn đầu được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Kết quả cho thấy chỉ 39/94 token, tương đương 41.49%, nằm trong loss. Phần system và user prompt bị mask, còn câu trả lời assistant được supervise. Điều này chứng minh mô hình đang học phần trả lời thay vì học lại hoặc sao chép prompt.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3164.6 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1039.5 |
| (c) LoRA fine-tune | 0.970 | 0.7444 | 1.000 | 1456.2 |

**(b) có thật sự mạnh hơn (a) không?** Có.

Baseline tối ưu tăng target từ 0.000 lên 0.765 và format từ 0.000 lên 1.000. Do đó đây là một baseline mạnh thực sự, không phải prompt yếu được tạo ra để làm fine-tune trông tốt hơn.

Tôi không sửa `OPTIMIZED_PROMPT` sau khi baseline được đóng băng. SHA được ghi nhận là `719e74d3b6232053`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | target | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6268 | 0.970 | 956.7 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5372 | 0.970 | 801.1 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 934.8 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1002.6 | 7.09 |

### 4.1 — Adapter placement và rank

`attn_only` được tăng rank lên 283 để có 32,456,704 tham số trainable, gần như bằng `correct` với 32,464,896 tham số. Trên tập target, hai cấu hình hòa nhau ở 0.970. Tuy nhiên training loss của `attn_only` là 0.5372, thấp hơn `correct` là 0.6268.

Nếu chỉ xếp hạng bằng training loss, tôi sẽ kết luận sai rằng `attn_only` tốt hơn. Kết quả này cho thấy với tác vụ triage JSON hẹp này, tăng rank để bù ngân sách tham số không tạo ra lợi thế target so với cấu hình text-linear. Rank và vị trí adapter phải được đánh giá bằng metric của tác vụ thay vì chỉ nhìn loss huấn luyện.

### 4.2 — Learning rate

`wrong_lr` chỉ thay learning rate từ 1e-4 xuống 1e-5 nhưng final loss tăng từ 0.6268 lên 1.5704. Quan trọng hơn, target của nó bằng 0.000 và format cũng bằng 0.000.

Như vậy learning rate quá thấp khiến LoRA không học đủ hành vi cần thiết trong cùng ngân sách 30 optimizer step. Nếu chỉ nhìn thấy loss có xu hướng giảm mà không kiểm tra target và format, tôi có thể nhầm rằng run vẫn đang học bình thường trong khi năng lực thực tế trên tác vụ gần như chưa hình thành.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ 12.01 GB xuống 7.09 GB, tương đương tiết kiệm khoảng 41%. Đổi lại target giảm từ 0.970 xuống 0.940 và thời gian train tăng từ 956.7 giây lên 1002.6 giây.

Trong thí nghiệm này, lợi ích về bộ nhớ là rõ ràng nhưng không miễn phí về chất lượng và cũng không giúp train nhanh hơn. Kết quả vì vậy ủng hộ việc thận trọng với QLoRA khi GPU vẫn đủ chạy LoRA 16-bit. Tuy nhiên mức giảm target chỉ 0.03 nên kết quả này không đủ để kết luận QLoRA luôn kém trong mọi bài toán.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`

`target Δ = +0.205`  
`regression Δ = -0.013`  
`valid_trace_rate = 0.00`

Fine-tune tăng target từ 0.765 của optimized-prompt baseline lên 0.970, tức tăng 0.205 tuyệt đối. Format vẫn giữ 1.000, cho thấy mức tăng target không đến từ việc đánh đổi định dạng JSON. Regression giảm từ 0.7578 xuống 0.7444, tương đương khoảng -0.013, nhỏ hơn rất nhiều so với phần cải thiện trên tác vụ mục tiêu.

Latency của fine-tune là 1456.2 ms, chậm hơn baseline tối ưu 1039.5 ms nhưng nhanh hơn đáng kể baseline naive 3164.6 ms. Vì vậy tôi chấp nhận phán quyết `PASSED`: fine-tune tạo ra cải thiện lớn trên tác vụ chuyên biệt mà chưa gây suy giảm đáng kể khả năng tổng quát theo metric regression của lab.

`valid_trace_rate=0.0` không được tôi dùng để kết luận reasoning collapse trong thí nghiệm chính này, bởi corpus mặc định huấn luyện bằng câu trả lời JSON chứ không có reasoning trace thực sự để đánh giá hiện tượng đó.

---

## 6. Định tính — có cả các ca fine-tune dự đoán sai

| # | Ticket rút gọn | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Trả chuột không dây, gấp | `doi_tra / cao` | `hoan_tien / cao` | `doi_tra / cao` | ✅ FT sửa đúng intent |
| 2 | Hoàn tiền ốp lưng, sớm nhé | `hoan_tien / trung_binh` | urgency=`cao` | urgency=`trung_binh` | ✅ FT sửa đúng urgency |
| 3 | Chưa thấy tiền, khi nào tiện | urgency=`thap` | urgency=`trung_binh` | urgency=`trung_binh` | ❌ FT vẫn sai urgency |
| 4 | Áo khoác bị lỗi, khi nào tiện | urgency=`thap` | urgency=`trung_binh` | urgency=`trung_binh` | ❌ FT vẫn sai urgency |
| 5 | Giao đèn LED chậm, khi nào tiện | `van_chuyen / thap` | `hoi_thong_tin / trung_binh` | `van_chuyen / trung_binh` | ⚠ FT sửa intent nhưng vẫn sai urgency |

Trên toàn bộ 50 target mẫu, fine-tune tốt hơn baseline (b) ở 33 mẫu, hòa ở 17 mẫu và không có mẫu nào có field-accuracy thấp hơn baseline. Vì vậy tôi không tạo ra các “ca fine-tune thua baseline” không tồn tại trong kết quả.

Tuy nhiên fine-tune vẫn có 6/50 mẫu chỉ đạt 0.75. Mẫu lỗi chung rất rõ là cách diễn đạt urgency mềm như “khi nào tiện”: nhãn đúng là `thap` nhưng fine-tune thường dự đoán `trung_binh`. Hai hàng được đánh dấu ❌ phía trên là failure case thực sự của fine-tune so với ground truth, dù chúng hòa baseline (b). Việc trình bày như vậy giữ nguyên tính trung thực của dữ liệu thay vì cherry-pick hoặc thay đổi tập đánh giá sau khi đã thấy kết quả.

---

## 7. Kết luận & điều tôi học được

Kết quả của thí nghiệm cho thấy bản LoRA fine-tune đáng cân nhắc triển khai cho tác vụ triage ticket cụ thể này. Bằng chứng mạnh nhất không phải training loss giảm mà là target tăng từ 0.765 của một optimized-prompt baseline đã được đóng băng trước khi training lên 0.970, trong khi format vẫn bằng 1.000 và regression chỉ giảm khoảng 0.013. Điều này cho thấy fine-tune đã học được hành vi chuyên biệt hữu ích thay vì chỉ làm thay đổi loss huấn luyện. Tuy nhiên tôi chưa nên deploy trực tiếp mà không kiểm thử thêm trên dữ liệu thực tế, đặc biệt với các cách diễn đạt urgency mềm như “khi nào tiện”, vì đây là nhóm lỗi lặp lại trong các mẫu chưa đạt điểm tuyệt đối.

Các thí nghiệm đối chứng cũng cho thấy không thể suy ra cấu hình tốt chỉ từ một hyperparameter hay một chỉ số thay thế. `attn_only` có training loss thấp hơn `correct` nhưng target chỉ hòa; `wrong_lr` thất bại hoàn toàn dù chỉ đổi learning rate; còn QLoRA tiết kiệm khoảng 41% VRAM nhưng target giảm 0.03. Trước tất cả các so sánh đó, loss mask đúng là điều kiện tiên quyết: nếu prompt cũng nằm trong loss thì những số liệu downstream sẽ không còn chứng minh đúng câu hỏi mà lab đặt ra. Bài học lớn nhất của tôi là phải chứng minh pipeline, đóng băng baseline và thiết kế đối chứng công bằng trước khi diễn giải kết quả cuối.

**Ba điều tôi học được:**

1. Training loss thấp hơn không đồng nghĩa target tốt hơn: `attn_only` có loss 0.5372 nhưng chỉ hòa `correct` ở target 0.970.
2. Learning rate là đòn bẩy rất mạnh đối với LoRA: giảm từ 1e-4 xuống 1e-5 làm target từ 0.970 xuống 0.000 trong cùng 30 step.
3. QLoRA tạo trade-off rõ ràng: VRAM giảm từ 12.01 GB xuống 7.09 GB nhưng target giảm từ 0.970 xuống 0.940.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** tăng số lượng và độ đa dạng của các mẫu urgency=`thap`, đặc biệt các cách diễn đạt gián tiếp như “khi nào tiện”, sau đó đánh giá trên một hold-out set mới chưa từng được xem trước. Tôi cũng muốn quét rank có kiểm soát trên cùng `text-linear` placement để tách riêng ảnh hưởng của rank khỏi ảnh hưởng của vị trí adapter.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
