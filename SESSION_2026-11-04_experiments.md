# Session log — 2026-11-04 (phiên thứ hai trong ngày)

Phiên này nối tiếp phiên "Gộp phần dữ liệu thành một notebook" cùng ngày (xem `SESSION_LOG.md`).
File này là **nhật ký đầy đủ của riêng phiên này**. Trạng thái hiện tại của project thì xem `SESSION_NOTES.md`.

Toàn bộ phiên do user dẫn dắt bằng câu hỏi, không theo kế hoạch có sẵn.

---

## Tóm tắt

| | Trước phiên | Sau phiên |
|---|---|---|
| Notebook | 2 (`01_data`, `02_model_training`) | **4** (thêm `03_direction_metrics`, `04_skip_connection`) |
| Notebook 02 | 29 cell, 3 mô hình gộp trong 1 hàm `build_model` | 31 cell, **mỗi mô hình một cell riêng** |
| Lần chạy 02 | Chưa chạy, output trống | **Đã chạy 5 seed**, mọi tài liệu cập nhật theo |
| Trang xuất bản | 2 | **3** |
| Kết luận chính | "Thua baseline 2,9–5,8 lần vì ngoại suy" (suy luận) | **Đo được trực tiếp**, và tách thành **hai** vấn đề riêng biệt |

Thành quả lớn nhất: chứng minh được bằng số rằng kết quả kém đến từ **cách đặt bài toán**, không phải từ
mô hình — và định lượng được cái giá của nó là **2,7 đến 5,3 lần sai số**.

---

## 1. Hỏi đáp mở đầu

**Trang Đọc hiểu code đã theo cấu trúc mới chưa?** Lúc đó đã khớp (bản 2). Sau khi sửa 02 ở mục 2 thì lệch,
nên phải dựng lại.

**Chồng thêm hidden layer hoặc Dense layer có tăng hiệu năng không?** Trả lời: gần như chắc chắn **không**,
vì nút thắt không nằm ở capacity:

1. Target nằm ngoài miền train (665/665 ngày test vượt 835,74 USD) — mạng sâu hơn vẫn phải ngoại suy.
2. Mọi lớp recurrent vẫn dùng `tanh`, chặn trong [−1, 1] — chồng lớp không bỏ được trần.
3. Train chỉ có 3.071 sequence — thêm tham số dễ overfit, early stopping sẽ dừng sớm.
4. LSTM hơn SimpleRNN nhờ cơ chế cổng, không phải nhờ sâu hơn: cả ba đều chỉ 1 lớp recurrent.

**Về sau phiên này đã kiểm chứng được dự đoán đó** bằng HP study (units 64→128 chỉ được 0,5 USD) và bằng
notebook 04 (sửa đúng chỗ ngoại suy thì sai số giảm 2,7–5,3 lần).

---

## 2. Viết lại notebook 02 — mỗi mô hình một cell

User thấy 02 rối vì nhiều code, muốn **mỗi mô hình có một cell riêng** thay vì `build_model(kind)` + dict
`LAYERS`. Kết quả: 29 → 31 cell, nhưng code tổng **ngắn hơn**.

| Thay đổi | Lý do |
|---|---|
| Mục 2 = quy trình train chung: `train`, `predict`, `train_seeds(name, build)` | Vòng 5 seed giống hệt nhau cho 3 mô hình, viết 1 lần |
| Mục 3 chia 3.1 / 3.2 / 3.3, mỗi cell tự viết `Sequential` + `compile` rồi gọi `train_seeds`, in `summary()` | Yêu cầu của user |
| `build_simplernn(input_shape, units, lr)` có tham số; GRU/LSTM không | Mục 9 (HP study) dùng lại SimpleRNN |
| `train_seeds` xóa run cũ của mô hình đó trước khi chạy | Chạy lại một cell mô hình không bị nhân đôi dòng |
| Bảng overfitting ở mục 10 **gộp vào bảng persistence ở mục 7** | Hai bảng gần trùng; dùng mean 5 seed thay vì seed tốt nhất |
| Bỏ bảng "days above train max" ở mục 10 | Trùng bảng tổng hợp của 01 |
| Bỏ hình gộp `test_predictions.png`, giữ hình 3 panel | Đường persistence đè lên giá thật, không đọc được |
| Mục 8 bỏ dòng format chuỗi `mean ± std` | Dòng đó dài và khó đọc |
| Mục 9 thay `study()` + lambda bằng danh sách `settings` và một vòng lặp | Dễ đọc hơn |
| Biến `runs` (list) không còn bị ghi đè thành DataFrame; DataFrame tên `results` | Ghi đè làm hỏng khi chạy lại cell |
| Tiêu đề "# 03 ·" → "# 02 ·", thêm bảng mục → requirement ở đầu | Sót từ lần đổi tên |

Số mục 4–11 **giữ nguyên** nên các con trỏ `02 mục x` trên trang Thẩm định vẫn đúng chỗ.

Đã sao lưu notebook vào scratchpad trước khi sửa (file chưa commit nên không có bản nào khác), rồi chạy thử
bản rút gọn (2 seed, `MAX_EPOCHS = 2`, output trỏ vào scratchpad): 1 phút 47 giây, không lỗi.

---

## 3. User chạy 02 — kiểm tra output và cập nhật mọi tài liệu

### 3.1 Kiểm tra

15 cell chạy liền mạch `execution_count` 1→15, không output `error` nào, 4 hình nhúng, 3 file `.keras` và
3 CSV mới. Mọi file trong `results/02_model_training/` mtime 09:50–10:06, trừ `test_predictions.png` còn từ
hôm trước — đúng là file thừa đã dự đoán, **đã xóa**.

### 3.2 Năm phát hiện

Ba cái đầu chỉ lộ ra khi chạy nhiều seed.

1. **Độ lệch giữa seed rất lớn** — test RMSE 120–280 (SimpleRNN), 122–187 (GRU), 72–181 (LSTM), tức
   2,3–2,5 lần trong cùng một kiến trúc. Quyết định chạy 5 seed ở phiên trước là đúng.
2. **Validation gần như không dự báo được test** — tương quan val/test RMSE trên 15 lượt chỉ **0,34**.
   Seed chọn theo val không phải seed tốt nhất trên test ở GRU và LSTM (LSTM: 99,2 so với 72,4).
3. **LSTM chạm trần 100 epoch ở 3/5 seed** (best epoch 93, 99, 99). Early stopping không kích hoạt.
4. **Underfitting, không phải overfitting** — cả 3 model thua persistence ngay trên train (7,60–7,97 so với
   6,39). Train→val tăng 2,3 lần, nhưng persistence cũng tăng 2,5 lần, nên mức tăng là của dữ liệu.
5. **Capacity không phải nút thắt** — units 16→128 cải thiện 28,88 → 17,50, nhưng bước 64→128 chỉ đáng
   0,5 USD. Window 10 (17,29 ± 0,25) tốt bằng hoặc hơn window 30 (18,02 ± 1,13).

**Hai chỗ đang sai đã phát hiện:** Discussion mục 10 viết "early stopping ends training well before 100
epochs" (sai với LSTM), và hình `test_predictions_per_model.png` dùng seed chọn theo val nên tiêu đề ghi
SimpleRNN R² 0,962 > GRU 0,958, **ngược với bảng** (205,34 so với 158,48).

### 3.3 Cập nhật

| Nơi | Thay đổi |
|---|---|
| `02` mục 5, 8, 10 + Discussion | Chỉ sửa **markdown**, output giữ nguyên. Discussion viết lại hoàn toàn, 8 ý, mỗi ý có số |
| `README.md` | Bảng Results mean ± std, thêm bảng RMSE theo từng tập, Findings 6 ý, HP study mean ± std |
| Trang Đọc hiểu code (bản 4) | Footer và mô tả mục 5 |
| Trang Thẩm định (bản 4) | 17 phép thay thế + thêm **phát hiện 9 và 10** + thay ảnh nhúng base64 |

Chi tiết khi sửa trang Thẩm định: biểu đồ tỉ lệ RMSE là **SVG viết tay**; tỉ lệ test mới 5,77 vượt khỏi
trục cũ (hết ở 5×), nên đổi thang 1× = 80px, trục tới 6×, và **sinh lại toàn bộ** gridline/tick/bar/dot/
label bằng vòng lặp Python thay vì sửa tay.

---

## 4. Câu hỏi về metric — hai hiểu nhầm

**Hỏi:** *"mô hình không sử dụng các metric thông thường để đánh giá đúng không?"*

Không đúng theo nghĩa đen. MAE/MSE/RMSE/R² **chính là** bộ metric chuẩn của hồi quy, và đúng bộ §7 yêu cầu.
Nhưng trực giác của user đúng ở chỗ khác, quan trọng hơn: **4 metric này nói rất ít về bài giá cổ phiếu**,
vì chúng chỉ đo "lệch bao nhiêu" trên chuỗi có xu hướng, nên ưu ái bất kỳ phương pháp nào bám theo đường
giá. R² = 0,968 nghe tốt, nhưng persistence đạt 0,9967.

**Hỏi tiếp:** *"vì sao bài này phải dùng các metric như trên mà không dùng loss, accuracy?"*

- **Loss vẫn đang được dùng.** `model.compile(..., loss="mse")`. Loss và metric là **hai vai trò**, không
  phải hai lựa chọn: loss cho optimizer (thang scaled, mỗi batch), metric cho người đọc (thang USD, sau khi
  train). Kiểm chứng: `√0,0005 × (835,74 − 50,06) ≈ 17,6 USD`, khớp val RMSE 17,00–18,75.
- **Accuracy không tồn tại cho hồi quy.** Đầu ra là số thực liên tục; giá thật 2.847,36 mà đoán 2.847,35
  thì accuracy tính là **sai**, ngang với đoán 50 USD. Thêm lý do kỹ thuật: accuracy là **hàm bậc thang**,
  đạo hàm 0 gần như mọi nơi, nên không bao giờ làm loss được.

Ranh giới cần nhớ: classification → nhãn rời rạc → "đúng hay sai?" → accuracy/F1/AUC. Regression (bài này)
→ số thực → "**lệch bao nhiêu?**" → MAE/MSE/RMSE/R².

Vì sao cần tới 4 metric: MAE dễ đọc và phạt đều; MSE phạt nặng lỗi lớn và khả vi mượt nên làm loss, nhưng
đơn vị USD² không đọc được; RMSE đưa về lại USD; R² không đơn vị nên so được giữa dataset. Đặt MAE cạnh
RMSE còn cho thêm thông tin: **102,97 / 74,71 = 1,38**, lớn hơn 1 đáng kể nghĩa là sai số **dồn vào một số
ngày**.

**Đối chiếu lại đề: không mục nào thiếu.** §7 đủ 4 metric trên USD; §8.3 so persistence trên cả 3 tập;
§8.4 so 4 metric giữa các mô hình. Đề **không cấm thêm** metric; §8.6 là thảo luận tự do.

---

## 5. Đề xuất tăng hiệu năng

Xếp theo hiệu quả thật. **Đề xuất 1 đã làm (notebook 04), đề xuất 4 đã làm (notebook 03). Đề xuất 2 và 3
chưa làm.**

**1 — Skip connection (ĐÃ LÀM, xem mục 7).** Thay đổi duy nhất thực sự gỡ được nút thắt, **vẫn giữ target
là giá đóng cửa** như §4.6:

```python
# thay vì:  y_pred = Dense(1)(h)
# làm:      y_pred = x[:, -1, close_idx] + Dense(1)(h)
```

Mạng chỉ phải học Δ, mà Δ **ổn định giữa ba tập** (std log-return 0,0196 / 0,0148 / 0,0194). Trần `tanh`
biến mất. **Sàn mới là persistence** thay vì 103–205.

**2 — Ensemble 5 seed (CHƯA LÀM).** Notebook 02 chạy 5 seed rồi **vứt đi 4**. Với độ lệch đã đo, lấy trung
bình dự báo của 5 seed gần như chắc chắn tốt hơn. Không phải train lại, nhưng notebook hiện **không lưu
`pred` ra file** nên phải sửa 02.

**3 — Nâng `MAX_EPOCHS` (CHƯA LÀM).** LSTM trong 02 dùng hết 100 epoch ở 3/5 seed. Dễ, cải thiện nhỏ.
(Trong 04 thì không còn vấn đề này: 11–54 epoch.)

**4 — Thêm metric (ĐÃ LÀM, notebook 03).**

**Những thứ KHÔNG nên kỳ vọng** — đã có dẫn chứng, đừng thử lại:
- *Thêm layer / Dense* — HP study: units 64→128 chỉ được 0,5 USD, trong nhiễu seed.
- *Thêm feature OHLCV* — Open/High/Low tương quan > 0,999 với Close, và cũng bị ngoại suy y hệt.
- *Window dài hơn* — window 60 (18,83) còn kém window 10 (17,29).

### Tham khảo bên ngoài (tra web trong phiên này)

Kết quả của nhóm **không bất thường**; đây là hiện tượng phổ biến và có tài liệu.

| Nguồn | Nội dung liên quan | Độ tin cậy |
|---|---|---|
| [Diff-LSTM, Springer 2026](https://link.springer.com/article/10.1007/s00521-026-12369-5) | Train trên chuỗi sai phân bậc 1 rồi dựng lại mức giá. Nêu **đúng** chẩn đoán của chúng ta: `tanh`/`sigmoid` bị chặn và bão hòa gây **thiên lệch xuống dưới** khi train trực tiếp trên chuỗi mức giá có xu hướng | Bài báo, **nguồn tốt nhất cho đề xuất 1** |
| [HuggingFace, stock-lstm-forecast](https://huggingface.co/shalev396/stock-lstm-forecast) | LSTM trên AAPL: RMSE test **4,483** so với **4,469** của persistence — làm cẩn thận cũng chỉ *hòa* | Project cá nhân |
| [Nature, 2025](https://www.nature.com/articles/s41599-025-04761-8) | Dự báo LSTM bám giá thật nhưng **trễ một ngày**; cảnh báo "độ chính xác 95–98%" là metric gây hiểu lầm | Peer-reviewed |
| [Research Square preprint](https://www.researchsquare.com/article/rs-8703148/v1) | Xử lý "persistence problem" bằng log-return: tự tương quan lag-1 giảm 0,89 → 0,23, nhưng DA chỉ **59,3%**, R² **0,16** | **Preprint**, chưa peer-review |
| [GitHub, ShivkumarDiwan](https://github.com/ShivkumarDiwan/Stock_Price_Prediction_using_LSTM) | Khuyến nghị đúng hai thứ project đang thiếu: so RMSE với naive baseline, và báo cáo directional accuracy | Repo cá nhân |
| [Medium, 14 models](https://medium.com/@umang.gulati19/the-verdict-what-14-models-taught-me-about-stock-prediction-13217024bac0) | Một transformer dồn **84,56%** attention vào giá hôm qua, tức sao chép baseline | Blog, **giai thoại** |

**Kỳ vọng hợp lý:** không phải "thắng persistence", mà **đưa mô hình về ngang persistence thay vì thua
5,8 lần** — đúng điều notebook 04 đạt được.

---

## 6. Notebook 03 — directional accuracy

`code/03_direction_metrics.ipynb`, 21 cell (11 code), chạy **18 giây** vì chỉ load 3 model đã lưu, 0 lỗi,
3 hình. Markdown tiếng Anh cho đồng bộ với 01 và 02.

| Mục | Nội dung |
|---|---|
| 1 | Setup; 1.1 load sequences + scaler + 3 model, assert hai scaler cùng thang |
| 2 | Directional accuracy, có công thức LaTeX; **2.1 chứng minh bằng code vì sao persistence không phải baseline** |
| 3 | Độ lệch hệ thống: bảng Δ dự báo TB và hình phân phối Δ |
| 4 | Có tín hiệu ẩn dưới độ lệch không: DA sau khử lệch + corr Pearson/Spearman + scatter |
| 5 | MAPE và Theil's U |
| 6 | Bảng tổng hợp → `direction_metrics.csv` |
| 7 | Discussion |

### Kết quả (seed được chọn theo validation, không phải mean 5 seed)

| | DA test | Dự báo "tăng" | DA sau khử lệch | corr(Δ) | MAPE | Theil's U |
|---|---|---|---|---|---|---|
| Persistence | 44,1 % | 0 % | — | — | 1,34 % | 1,00 |
| SimpleRNN | 45,1 % | 3,5 % (23/665) | 51,6 % | 0,037 (p 0,34) | 3,82 % | 3,38 |
| GRU | 46,5 % | 4,8 % (32/665) | 51,6 % | 0,038 (p 0,33) | 3,91 % | 3,55 |
| LSTM | 47,2 % | 6,2 % (41/665) | 52,5 % | 0,052 (p 0,18) | 3,16 % | 2,79 |
| *Luôn "tăng"* | *55,9 %* | *100 %* | | | | |

Trên validation: DA 47,2–53,1 %, Theil's U 1,01–1,06, corr −0,018 đến 0,008, **không giá trị nào có ý nghĩa
thống kê**.

### Ba phát hiện

1. **DA test 45–47 %, dưới cả tung đồng xu**, và xa mức 55,9 % của "luôn đoán tăng".
2. **Nguyên nhân là độ lệch hệ thống, không phải tín hiệu ngược.** Mô hình dự báo giảm **64–83 USD mỗi
   ngày** trong khi thực tế tăng 2,5 USD/ngày → đoán "giảm" ở **94–96 %** số ngày test. Đảo ngược dự báo
   cũng vô ích vì sẽ thành "luôn đoán tăng" = majority class.
3. **Khử lệch xong thì không còn gì.** DA về 48,7–52,5 %; corr từ −0,018 đến 0,055; **không một giá trị nào
   trong 12 giá trị có ý nghĩa thống kê** (p từ 0,18 đến 0,84).

### Kết luận lớn nhất — tách hai vấn đề mà 4 metric bắt buộc gộp làm một

| Vấn đề | Bằng chứng | Thấy ở đâu | Sửa được? |
|---|---|---|---|
| Target ra ngoài miền train | Theil's U 2,8–3,5; Δ dự báo TB −64 đến −83 USD | **Chỉ test** | **Được** |
| Chuỗi gần bước ngẫu nhiên | DA bằng tung đồng xu, corr ≈ 0 | **Cả val lẫn test** | **Không** |

Bằng chứng nằm ở validation: ở đó **gần như không có độ lệch** (Δ dự báo TB −1,1 đến +0,9 so với thực tế
+0,5), Theil's U chỉ 1,01–1,06 tức gần hòa baseline về RMSE — **vậy mà DA vẫn là tung đồng xu**.

---

## 7. Notebook 04 — skip connection

`code/04_skip_connection.ipynb`, 24 cell (13 code), chạy **~13 phút** (15 lượt train), 0 lỗi, 3 hình.
Ghi `models/skip_*.keras` và `results/04_skip/`.

Phải dùng functional API vì `Sequential` không định tuyến được input vòng qua lớp recurrent:

```python
inp   = Input(shape=(30, 1))
h     = Dropout(0.2)(layer(units, activation="tanh")(inp))
delta = Dense(1, name="delta")(h)
last  = inp[:, -1, CLOSE_IDX:CLOSE_IDX + 1]   # (batch, 1)
out   = Add()([last, delta])
```

### Kết quả

| Model | NB 02 | NB 04 (skip) | Cải thiện | So baseline |
|---|---|---|---|---|
| SimpleRNN | 205,34 ± 75,75 | **38,47 ± 3,59** | 5,3× | 1,08× |
| GRU | 158,48 ± 31,79 | **38,95 ± 4,49** | 4,1× | 1,10× |
| LSTM | 102,97 ± 45,28 | **37,51 ± 2,48** | 2,7× | 1,05× |
| Persistence | 35,57 | 35,57 | — | 1,00× |

- **Chẩn đoán được xác nhận.** Khoảng cách tới baseline: 2,90–5,77× → **1,05–1,10×**. Mọi thứ khác giữ
  nguyên nên nguyên nhân cũ chỉ có thể là cái đã gỡ bỏ. **Capacity chưa bao giờ là nút thắt.**
- **Độ lệch giữa seed sụp đổ:** ±75,75/±31,79/±45,28 → ±3,59/±4,49/±2,48. Ba kiến trúc giờ gần như không
  phân biệt được nhau.
- **Train và val hoà hẳn baseline** (train 6,40–6,57 so 6,39; val 16,19–16,38 so 16,19). Underfitting ở 02
  biến mất.
- **Vẫn không thắng baseline.** Rõ nhất: GRU seed 45 dừng ở **epoch 1**, đạt 35,564 so với 35,566 của
  baseline — mạng thấy đoán "không đổi" là tốt ngang mọi thứ nó học được.
- Hội tụ nhanh hơn hẳn: **11–54 epoch** (02 là 22–100, LSTM chạm trần).
- **Không tạo ra kỹ năng đoán hướng**, đúng dự đoán của 03. DA test vẫn 42,7–44,4 %, corr −0,015 đến 0,021
  (p 0,60–0,88). Còn sót lệch nhỏ: −8 đến −12 USD/ngày, **giảm 7 lần** so với −64…−83 ở notebook 03.

### BẪY QUAN TRỌNG NHẤT CỦA PHIÊN — đừng đọc nhầm con số validation của 04

Trên val, corr là **0,088–0,107 với p = 0,006–0,023**, nhìn như "có ý nghĩa thống kê". Suýt nữa thì báo cáo
đây là bằng chứng mô hình học được điều gì đó.

**Không được tin**: chính validation đã *chọn ra* các model này — early stopping chọn epoch theo val loss,
notebook chọn seed theo val loss. Metric tính trên dữ liệu dùng để chọn thì **luôn bị thổi phồng**.
**Test là ước lượng sạch duy nhất, và nó không cho thấy gì.**

Đây đúng là cái bẫy project đã tránh được ở mọi chỗ khác bằng cách không bao giờ chọn gì theo test. Đã nói
thẳng ở notebook 04 mục 7, README và trang mới.

---

## 8. Ba trang đã xuất bản

Mỗi trang có **một đối tượng đọc khác nhau** — đừng gộp nội dung giữa chúng.

| Trang | Link | Chia sẻ | Viết cho ai |
|---|---|---|---|
| Thẩm định pipeline (bản 4) | https://claude.ai/artifact/SDqYDmZePgebmz9MmQULtT | anyone with the link | Người chấm / review kỹ thuật |
| Đọc hiểu code (bản 5) | https://claude.ai/artifact/S29cQ3jSfvgLmXRypbDw1v | riêng tư | Người cần đọc hiểu code |
| Hai thí nghiệm 03 + 04 (mới) | https://claude.ai/artifact/AAPr3JDeJhbhqZsXCDATzt | riêng tư | Người cần hiểu để **trình bày**, không đọc code |

### Trang Đọc hiểu code — bản 5

User phản hồi phần mô hình RNN "hơi vắn tắt và dễ gây khó hiểu". Đã thêm hẳn **một mục 3 mới** ("Ba mô hình
hoạt động thế nào") đặt trước 3.1:

- **Sơ đồ SVG RNN trải phẳng** — 4 ô thời gian, mũi tên bộ nhớ chạy ngang, input đi lên từ dưới, cuối cùng
  vào Dense(1). Chú thích "cả 4 ô dùng chung một bộ trọng số". Màu dùng `var(--accent)` / `var(--muted)`
  nên tự hợp cả light lẫn dark mode.
- Công thức `h_t = tanh(W_x·x_t + W_h·h_{t−1} + b)` kèm **bóc tách 4.224 tham số** = 64 + 4.096 + 64.
- **Bảng hành trình shape** (32,30,1) → (32,64) → (32,64) → (32,1).
- **Hộp cảnh báo nối `tanh` với vấn đề chính của bài** — chỗ quan trọng nhất: giải thích kiến trúc xong thì
  chỉ thẳng ra rằng chính `tanh` gây ra trần, và notebook 04 sửa đúng chỗ đó.
- Bảng so SimpleRNN / GRU / LSTM kèm giải thích "cổng là gì" bằng ẩn dụ cái van.

Ngoài ra mở rộng 6 chú thích vắn tắt (Dropout, Dense(1), GRU gates, LSTM cell state, units/tanh, Input
shape). Thêm CSS cho `.explain`, `.formula`, `.shape-tbl`, `.diagram`, `.warn-box`, `.paramcalc`.

### Trang thứ ba — mới tạo

Giải thích hai thí nghiệm bằng **tiếng Việt**, viết cho người cần hiểu để trình bày chứ không phải người
đọc code. Cấu trúc: tóm tắt 30 giây → vì sao cần 2 notebook (có sơ đồ mạch suy nghĩ 02 → chẩn đoán → 03 →
04) → notebook 03 → notebook 04 → tổng kết → hình.

Bảng màu **nâu/kem** để phân biệt với hai trang kia (Thẩm định xanh dương, Đọc hiểu xanh lá). Nhúng 3 hình
base64, tổng 305 KB.

Ví dụ dùng xuyên suốt để giải thích directional accuracy: *hôm nay 2.000, đoán 1.930, thật 2.005* — RMSE
chấm "khá tốt", directional accuracy chấm "sai hoàn toàn".

---

## 9. Bẫy kỹ thuật gặp trong phiên này

1. **Tên lớp Keras bị tự đánh số.** `probe.get_layer("dense")` fail với `No such layer: dense. Existing
   layers are: [... 'dense_1', 'add_1']` vì đã tạo model khác trước đó trong cùng session. **Sửa: đặt tên
   tường minh** `Dense(1, name="delta")`.
2. **Slicing KerasTensor hoạt động và serialize được.** `inp[:, -1, i:i+1]` cho shape `(batch, 1)` và
   `save()`/`load_model()` roundtrip đúng. **Không cần `Lambda`** (vốn rắc rối khi load với `safe_mode`).
   Dùng `i:i+1` chứ không phải `i`, nếu không ra shape `(batch,)` và `Add()` báo lỗi.
3. **Heredoc trong Bash hỏng khi script Python có `'''` hoặc `\n`.** Gặp 2 lần: `'''` trong heredoc
   `<< 'PYEOF'` làm shell báo `unexpected EOF`; dùng heredoc để *vá* script thì `\\n` bị nuốt thành xuống
   dòng thật, gây `SyntaxError`. **Dùng Write tool để tạo script, Edit tool để sửa.**
4. **Assert "không còn chuỗi cũ" phải loại trừ mục nhật ký.** `SESSION_NOTES.md` mục 8 ghi lại trạng thái
   tại thời điểm đó nên vẫn chứa "02 chưa chạy", và đó là đúng.
5. **Khung HTML do publish tự chèn.** Bản `read` về có `<!doctype …><body>` ở đầu và `</body></html>` ở
   cuối; trang Đọc hiểu code còn có bundle hljs (`<!--claude-hljs-runtime-begin…end-->`). Khi dựng lại
   **phải cắt cả hai đầu**, nếu không trang sẽ có 2 khung.
6. **Persistence không dùng được làm baseline cho directional accuracy.** Δ của nó luôn bằng 0 nên suy biến
   thành "luôn đoán giảm", điểm bằng đúng tỉ lệ ngày giảm. Baseline đúng là majority class.
7. **Hai scaler phải cùng thang.** Phép so hướng lấy `y` (thang `target_scaler`) trừ ngày cuối window
   (thang `feature_scaler`). Notebook 03 assert `data_min_`/`data_max_` bằng nhau — đúng vì
   `FEATURES == [TARGET] == ["Close"]`. Nếu bật multivariate thì phải kiểm tra lại `close_idx`.
8. **Khử lệch bằng median, không phải mean**, vì phân phối Δ dự báo lệch (mean −80,60 so với median −39,07).
9. **Dùng Spearman song song Pearson** vì thay đổi giá hằng ngày có đuôi nặng.
10. Chạy nbconvert ở **background** cho notebook dài, rồi sửa markdown Discussion sau khi có số thật —
    sửa markdown không làm mất output.

---

## 10. Script đã dùng (nằm trong scratchpad của phiên này, **đã mất**)

| Script | Việc |
|---|---|
| `rebuild02.py` | Viết lại `02_model_training.ipynb`: tách 3 mô hình ra 3 cell, giữ `id` của cell không đổi |
| `update_md02.py` | Viết lại 4 cell markdown của 02 theo số lần chạy 5 seed, assert output không mất |
| `build_page02.py` | Dựng lại article 02 của trang Đọc hiểu code từ notebook, assert mọi khối code khớp |
| `fix_report2.py` | 17 phép thay thế trên trang Thẩm định + sinh lại SVG + nhúng lại ảnh base64 |
| `dir_acc.py`, `dir_acc2.py` | Thí nghiệm directional accuracy trước khi tách thành notebook 03 |
| `build03.py` | Dựng `03_direction_metrics.ipynb` từ đầu bằng JSON |
| `probe.py` | Kiểm tra slicing KerasTensor + save/load trước khi dựng notebook 04 |
| `build04.py`, `fix04md.py` | Dựng `04_skip_connection.ipynb`; viết lại Discussion sau khi có số thật |
| `expand_rnn.py`, `expand_notes.py` | Thêm mục 3 (sơ đồ RNN) và mở rộng 6 chú thích trên trang Đọc hiểu code |
| `build_exp_page.py` | Dựng trang thứ ba từ đầu, đọc số trực tiếp từ các CSV trong `results/` |
| `upd_notes.py`, `upd_notes2.py` | Cập nhật `SESSION_NOTES.md` theo lần chạy 5 seed, rồi theo notebook 04 |
| `expand_log.py`, `collapse_log.py` | Mở rộng mục nhật ký, rồi gộp 4 mục trong `SESSION_LOG.md` thành con trỏ sang file này |

**Cách làm lại khi cần:** `read` artifact → bản đầy đủ lưu vào `tool-results/` → viết script thay thế chuỗi
trên bản đó → publish lại bằng cùng `url`. Với notebook thì sửa trực tiếp file `.ipynb` bằng script JSON.

---

## 11. Còn lại sau phiên này

1. Trang **Thẩm định pipeline** chưa nhắc tới notebook 03 và 04.
2. Đề xuất 2 (ensemble 5 seed) và 3 (nâng `MAX_EPOCHS` cho 02) chưa làm.
3. README còn thiếu mục `slides/` (folder chưa tồn tại).
4. **Phần II (slide Attention) vẫn chưa bắt đầu** — mới có 1 slide mẫu, đề yêu cầu 15–20 slide, 12 mục nội
   dung, 5 câu MCQ. **Đây là việc lớn nhất còn lại.**
5. **Chưa commit gì.**
