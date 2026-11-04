# Session notes — trạng thái project

File này để các session sau (người hoặc Claude) đọc nhanh **trạng thái** và tiếp tục.
**Mỗi session nên cập nhật mục "Trạng thái hiện tại" và "Việc còn lại", và thêm một dòng vào "Nhật ký".**

> Cần **chi tiết** hơn (đã làm gì, quyết định vì sao, bẫy kỹ thuật, cách đọc/sửa 3 trang đã xuất bản)
> thì xem `SESSION_LOG.md` — đó là **mục lục các phiên**, mỗi phiên một mục tóm tắt.
> Phiên dài có file riêng: hiện có [`SESSION_2026-11-04_experiments.md`](SESSION_2026-11-04_experiments.md)
> (viết lại 02, chạy 02, notebook 03 + 04, ba trang xuất bản, đề xuất cải tiến chưa làm, nguồn tham khảo).
> Mỗi session mới phải thêm một mục vào `SESSION_LOG.md`, và nếu dài thì tạo file riêng.

Cập nhật lần cuối: 2026-11-04

---

## 1. Bối cảnh

- Môn Deep Neural Networks, Midterm Project 2, Nhóm 1 (5 thành viên, xem README).
- Đề bài: file PDF *Midterm Requirements 2* (không nằm trong repo). Gồm hai phần:
  - **Phần I – RNN:** dự báo giá đóng cửa GOOGL ngày kế tiếp từ 30 ngày trước. Mục bắt buộc:
    §4.1–4.7 tiền xử lý (Min-Max **fit trên train**, window 30, target = Close ngày kế tiếp, kiểm tra leakage),
    §5 chia 70/15/15 theo thời gian, §6.1 `Input → SimpleRNN(64, tanh) → Dropout(0.2) → Dense(1)`, Adam, MSE, early stopping,
    §6.2 GRU/LSTM (tùy chọn), §7 MAE/MSE/RMSE/R² sau inverse transform,
    §8.1–8.6 plot dự báo, loss curves, so persistence, so 4 metric, thảo luận sequence length / units / lr / epochs, overfitting / underfitting / limitations.
  - **Phần II – Research, Topic 2 Attention:** 15–20 slide, thuyết trình 15–20 phút, 12 mục nội dung
    (giới thiệu, Attention trong Encoder–Decoder, Q/K/V, công thức, các loại attention, self-attention, multi-head,
    attention trong Transformer, độ phức tạp, ứng dụng, ưu/nhược so với RNN/LSTM, **5 MCQ: 2 dễ, 2 trung bình, 1 khó**).
- Dataset: `data/GOOGL.csv`, 4.431 dòng, 2004-08-19 → 2022-03-24, không có giá trị thiếu, `Close == Adj Close`.

## 2. Cấu trúc code (chỉ còn **2 notebook** sau khi gộp ngày 2026-11-04)

| Notebook | Vai trò | Đọc | Ghi |
|---|---|---|---|
| `code/01_data.ipynb` | Khám phá **và** tiền xử lý trong cùng một file: requirement §4.1–4.7 rồi §5, đúng thứ tự PDF | `data/GOOGL.csv` | `data/processed/sequences_w{10,20,30,60}.npz`, `feature_scaler.joblib`, `target_scaler.joblib`, `results/01_data/*.png` |
| `code/02_model_training.ipynb` | Requirement §6, §7, §8 | `data/processed/` | `models/{simplernn,gru,lstm}.keras`, `results/02_model_training/*` (gồm `runs_per_seed.csv`, `test_metrics.csv`, `hyperparameter_study.csv`) |
| `code/03_direction_metrics.ipynb` | **Thêm ngoài đề**, hỗ trợ §8.6: directional accuracy, MAPE, Theil's U | `data/processed/`, `models/*.keras` | `results/03_direction/*` (3 hình + `direction_metrics.csv`) |
| `code/04_skip_connection.ipynb` | **Thêm ngoài đề**, hỗ trợ §8.6: skip connection — dự báo độ lệch thay vì mức giá | `data/processed/`, `results/02_model_training/runs_per_seed.csv` | `models/skip_*.keras`, `results/04_skip/*` (3 hình + 3 CSV) |

- Tiêu đề mục trong notebook đánh số **1, 2, 3…** (theo yêu cầu của user). Khi trích đề thì viết rõ "requirement §x.y".
- 01 có 10 mục, 39 cell (17 code / 22 markdown). Mục có nhiều cell code thì chia tiếp `### x.y` để thấy rõ ranh giới từng cell. Map mục → đề: 2–4 = §4.1–4.3, 5 = §4.4, 6 = §4.5, 7 = §4.6, 8 = §4.7, 9 = §5, 10 = ghi file.
- **Không còn code trùng lặp**: trước đây 01 và 02 cùng load + sort + check missing. User yêu cầu bỏ phần lặp, và vì dữ liệu đã xác nhận sạch (0 missing) nên đoạn `ffill()` / `dropna()` **đã bị xóa hẳn**, thay bằng markdown giải thích rằng §4.3 không cần imputation. Nếu đổi sang dataset khác thì phải thêm lại.
- 01: sequence được gán vào train/val/test theo **ngày target**, nên val và test giữ đủ 665 mẫu. Input của các window đầu val/test lùi về split trước, hợp lệ vì là quá khứ.
- 01: `train_end` được tính ở mục 6 (normalize) vì scaler cần biên train trước, `val_end` ở mục 9. Markdown đã nói rõ lý do.
- 02 (sửa 2026-11-04): 11 mục, 31 cell (15 code). **Mỗi mô hình một cell riêng** ở mục 3 (3.1 SimpleRNN, 3.2 GRU, 3.3 LSTM), mỗi cell tự viết `Sequential([...])` + `compile`. Phần dùng chung nằm ở mục 2: `train`, `predict`, `train_seeds(name, build)`. `build_simplernn(input_shape, units, lr)` có tham số vì mục 9 dùng lại. Đầu file có bảng mục → requirement như 01.
- 02: mỗi kiến trúc train **5 seed** (`SEEDS = [42..46]`). Seed vẽ hình và lưu model được chọn theo **validation loss**. Kết quả báo cáo mean ± std. HP study dùng 3 seed cho mỗi cấu hình, có đường persistence trên biểu đồ. Bỏ `shuffle=False`, dùng shuffle mặc định của Keras.

## 3. Trạng thái hiện tại

- **01: đã chạy lại sau khi gộp, chạy hết không lỗi, output và hình đã nhúng trong notebook.** Mọi assert pass. Số liệu khớp với mục 4 dưới đây: train max 835,74 · 627/665 ngày val và 665/665 ngày test vượt mức đó · scaled Close max 1,586 (val) và 3,751 (test) · `X=(4401, 30, 1)`. `results/01_data/*.png` và `data/processed/*` là của lần chạy này.
- **02: user đã chạy xong ngày 2026-11-04 (5 seed).** 15 cell chạy liền mạch 1→15, không lỗi, 4 hình nhúng. `results/02_model_training/*` và `models/*.keras` là của lần chạy này. File thừa `test_predictions.png` (hình gộp của code cũ) đã xóa.
- README, phần Discussion ở 02 mục 10, và **cả hai trang đã xuất bản** đều đã cập nhật theo số của lần chạy này.
- **03: đã chạy, 11 cell, 0 lỗi, 3 hình, 18 giây.** Không train lại gì, chỉ load 3 model đã lưu. **Chạy rất nhanh nên cứ chạy lại thoải mái.** Lưu ý: chỉ đo trên seed được chọn theo validation của mỗi kiến trúc (chỉ 3 model đó được lưu), chưa có mean ± std qua 5 seed.
- **04: đã chạy, 13 cell, 0 lỗi, 3 hình, ~13 phút** (15 lượt train). Kết quả ở mục 4 dưới.
- **Đã có 3 trang xuất bản** (xem mục 7). Trang Thẩm định **chưa nhắc tới notebook 03 và 04**.
- **Chưa commit gì.** Branch `main`, HEAD `c41b618 Complete README`. Thay đổi chưa commit: README, SESSION_NOTES, `code/01_data.ipynb` (file mới), `code/02_model_training.ipynb` (đổi tên từ 03), xóa `01_data_explore.ipynb` + `02_preprocessing.ipynb`, `results/` đổi tên 2 folder, `data/processed/*`.
- Phần II: mới có `Claude outputs/sample_slide.pptx` (1 slide về scaled dot-product attention). Folder `slides/` mà README nhắc tới chưa tồn tại.

## 4. Kết quả và kết luận chính (lần chạy 2026-11-04, 5 seed, mean ± std)

| Model | Test MAE | Test RMSE | R² | Train RMSE | Val RMSE |
|---|---|---|---|---|---|
| Persistence | 24,79 | 35,57 | 0,9967 | 6,39 | 16,19 |
| SimpleRNN | 147,07 ± 52,88 | 205,34 ± 75,75 | 0,877 ± 0,078 | 7,97 | 18,01 |
| GRU | 115,17 ± 24,57 | 158,48 ± 31,79 | 0,932 ± 0,026 | 7,60 | 17,00 |
| LSTM | 74,71 ± 33,39 | 102,97 ± 45,28 | 0,968 ± 0,031 | 7,87 | 18,75 |

Train/val RMSE là trung bình 5 seed, notebook tính trực tiếp trên USD (không còn suy ra từ `val_loss`).
Tỉ lệ so với persistence: val 1,05–1,16 · test 2,90–5,77.

HP study (SimpleRNN, val RMSE, 3 seed mỗi cấu hình; persistence = 16,19):
units 16/32/64/128 = 28,88 / 19,71 / 18,02 / 17,50 · lr 0,01/0,001/0,0001 = 36,35 / 18,02 / 30,32 ·
epochs 5/20/60 (không early stop) = 52,78 / 28,48 / 24,00 · window 10/20/30/60 = 17,29 / 18,64 / 18,02 / 18,83.
**Không cấu hình nào thắng persistence.**

- Pipeline **đáp ứng đủ mọi mục bắt buộc** của đề. Không có leakage.
- Không model nào thắng persistence, kể cả trên validation.
- Nguyên nhân chính là **giới hạn của cách đặt bài toán theo đề**, không phải lỗi code: target là mức giá, Min-Max fit trên train, train max = 835,74 USD. **627/665** ngày val và **665/665** ngày test cao hơn mức đó (scaled val tới 1,586, test tới 3,751), nên mô hình phải ngoại suy. `tanh` chặn trạng thái ẩn trong [−1, 1], nên đầu ra có trần và dự báo tụt dưới giá thật khi giá tăng.
- Nguyên nhân phụ là bản chất của giá cổ phiếu, gần với bước ngẫu nhiên. Kiến trúc chỉ quyết định mức độ thua: LSTM ít nhất, SimpleRNN nhiều nhất.
- **Phát hiện mới từ lần chạy 5 seed** (đã đưa vào 02 mục 10, README và trang Thẩm định):
  1. **Độ lệch giữa seed rất lớn:** test RMSE trải 120–280 (SimpleRNN), 122–187 (GRU), 72–181 (LSTM).
  2. **Validation gần như không dự báo được test:** tương quan val–test RMSE trên 15 lượt chỉ **0,34**. Seed chọn theo val không phải seed tốt nhất trên test ở GRU và LSTM (LSTM: 99,2 so với 72,4).
  3. **LSTM chạm trần 100 epoch ở 3/5 seed** (best epoch 93, 99, 99) — early stopping không kích hoạt, `MAX_EPOCHS` đang là ràng buộc.
  4. **Underfitting, không phải overfitting:** cả 3 model thua persistence ngay trên train (7,60–7,97 so với 6,39). Train→val tăng 2,3 lần nhưng persistence cũng tăng 2,5 lần, nên mức tăng đó là của dữ liệu.
  5. **Capacity không phải nút thắt:** units 64→128 chỉ được 0,5 USD, nằm trong nhiễu seed. Window 10 (17,29) tốt bằng hoặc hơn window 30 (18,02).
  6. Hình `test_predictions_per_model.png` dùng seed chọn theo val, nên R² trên tiêu đề **không phải xếp hạng kiến trúc** (SimpleRNN trông hơn GRU, thực tế kém 47 USD). README và notebook đã ghi chú.

### Kết quả notebook 04 (skip connection) — 5 seed đầy đủ

Đổi đúng một dòng: `dự báo = giá hôm nay + Dense(1)(h)` thay vì `dự báo = Dense(1)(h)`. Target vẫn là giá
đóng cửa (§4.6), dữ liệu/split/scaler/seed/quy trình train giữ nguyên tuyệt đối.

| Model | NB 02 | NB 04 (skip) | Cải thiện | So baseline |
|---|---|---|---|---|
| SimpleRNN | 205,34 ± 75,75 | **38,47 ± 3,59** | 5,3× | 1,08× |
| GRU | 158,48 ± 31,79 | **38,95 ± 4,49** | 4,1× | 1,10× |
| LSTM | 102,97 ± 45,28 | **37,51 ± 2,48** | 2,7× | 1,05× |
| Persistence | 35,57 | 35,57 | — | 1,00× |

- **Chẩn đoán được xác nhận.** Khoảng cách tới baseline: 2,90–5,77× → **1,05–1,10×**. Mọi thứ khác giữ
  nguyên nên nguyên nhân cũ chỉ có thể là cái đã gỡ bỏ. **Capacity chưa bao giờ là nút thắt.**
- **Độ lệch giữa seed sụp đổ:** ±75,75/±31,79/±45,28 → ±3,59/±4,49/±2,48. Ba kiến trúc giờ gần như không
  phân biệt được.
- **Train và val hoà hẳn baseline** (train 6,40–6,57 so 6,39; val 16,19–16,38 so 16,19). Underfitting ở 02 biến mất.
- **Vẫn không thắng baseline.** Rõ nhất: GRU seed 45 dừng ở **epoch 1**, đạt 35,564 so với 35,566 của
  baseline — mạng thấy đoán "không đổi" là tốt ngang mọi thứ nó học được.
- Hội tụ nhanh hơn hẳn: **11–54 epoch** (02 là 22–100, LSTM chạm trần).
- **Không tạo ra kỹ năng đoán hướng**, đúng dự đoán của 03. DA test vẫn 42,7–44,4 %, corr −0,015 đến 0,021
  (p 0,60–0,88). Còn sót lệch nhỏ: −8 đến −12 USD/ngày (giảm **7 lần** so với −64…−83 ở 03).

**BẪY QUAN TRỌNG — đừng đọc nhầm con số validation của 04.** Trên val, corr là 0,088–0,107 với p
0,006–0,023, nhìn như "có ý nghĩa thống kê". **Không được tin**: chính val đã *chọn ra* các model này
(early stopping chọn epoch theo val loss, notebook chọn seed theo val loss). Metric tính trên dữ liệu dùng
để chọn thì luôn bị thổi phồng. **Test là ước lượng sạch duy nhất, và nó không cho thấy gì.** Notebook 04
mục 7 và README đều đã nói thẳng điều này.

### Kết quả notebook 03 (directional accuracy) — seed được chọn, không phải mean 5 seed

| | DA test | Dự báo "tăng" | DA sau khử lệch | corr(Δ) | MAPE | Theil's U |
|---|---|---|---|---|---|---|
| Persistence | 44,1 % | 0 % | — | — | 1,34 % | 1,00 |
| SimpleRNN | 45,1 % | 3,5 % | 51,6 % | 0,037 (p 0,34) | 3,82 % | 3,38 |
| GRU | 46,5 % | 4,8 % | 51,6 % | 0,038 (p 0,33) | 3,91 % | 3,55 |
| LSTM | 47,2 % | 6,2 % | 52,5 % | 0,052 (p 0,18) | 3,16 % | 2,79 |
| *Luôn "tăng"* | *55,9 %* | *100 %* | | | | |

Trên validation: DA 47,2–53,1 %, Theil's U 1,01–1,06, corr −0,018 đến 0,008, **không giá trị nào có ý nghĩa thống kê**.

**Kết luận quan trọng nhất của 03 — tách hai vấn đề mà 4 metric bắt buộc gộp làm một:**

| Vấn đề | Bằng chứng | Thấy ở đâu | Sửa được? |
|---|---|---|---|
| Target ra ngoài miền train | Theil's U 2,8–3,5; Δ dự báo TB −64 đến −83 USD | **Chỉ test** | Được, nếu dự báo độ lệch thay vì mức giá |
| Chuỗi gần bước ngẫu nhiên | DA bằng tung đồng xu, corr ≈ 0 | **Cả val lẫn test** | Không — bản chất dữ liệu |

Điểm mấu chốt: trên validation gần như **không có độ lệch** (Δ dự báo TB −1,1 đến +0,9 so với thực tế +0,5) và Theil's U chỉ 1,01–1,06, tức gần hòa baseline về RMSE — vậy mà **vẫn không có kỹ năng về hướng**. Nghĩa là sửa được ngoại suy sẽ cải thiện RMSE nhiều, nhưng **không tạo ra kỹ năng dự báo hướng**.

Lý do persistence **không phải** baseline cho DA: Δ của nó luôn đúng bằng 0 nên không bao giờ "tăng", thành ra suy biến thành "luôn đoán giảm" và điểm của nó chính là tỉ lệ ngày giảm. Baseline đúng là tỉ lệ ngày tăng (55,9 % test, 53,1 % val).
- Log-return ổn định giữa ba tập (std 0,0196 / 0,0148 / 0,0194), nên có thể dùng làm **thí nghiệm phụ** để chứng minh nguyên nhân. **Không** thay pipeline chính, vì đề bắt buộc target là giá đóng cửa.

## 5. Việc còn lại

1. Cập nhật trang **Thẩm định pipeline** để nhắc tới notebook 03 và 04 (hai trang kia đã xong).
2. (Cân nhắc) Nâng `MAX_EPOCHS` cho LSTM trong **02** rồi chạy lại: 3/5 seed dùng hết 100 epoch mà vẫn đang cải thiện. Trong 04 thì không còn vấn đề này (11–54 epoch).
3. (Cân nhắc) Ensemble 5 seed — 02 chạy 5 seed rồi vứt 4. Lấy trung bình dự báo gần như chắc chắn tốt hơn. Cần sửa 02 để lưu `pred`.
4. README còn thiếu mục `slides/` (folder chưa tồn tại).
5. (Tùy chọn) Notebook 05: thí nghiệm log-return, `Ĉ_t = C_{t−1}·exp(r̂_t)`, MinMax fit trên return của train.
6. **Phần II: làm bộ slide Attention đủ 12 mục cùng 5 MCQ.** Đây là việc lớn nhất còn lại — mới có 1 slide mẫu, đề yêu cầu 15–20 slide.
7. Commit khi user yêu cầu.

## 6. Ghi chú cho session sau

- **Môi trường Python:** dùng `python` (3.14 tại `C:\Users\slimz\AppData\Local\Python\bin`). Pandas, scikit-learn, TensorFlow 2.22, nbconvert nằm trong **user-site**, nên **không** được chạy với cờ `-I`, vì cờ này bỏ qua user-site và gây lỗi giả "không có pandas". Thư mục anaconda3 bị hỏng, không dùng.
- Chạy notebook: `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <file>.ipynb`, chạy từ trong `code/`, vì đường dẫn trong notebook là tương đối (`../data/...`).
- User muốn: trả lời bằng tiếng Việt; bám đúng đề PDF; **không tự chạy notebook dài khi chưa được yêu cầu**; giải thích dễ hiểu, không quá dài.
- Các script dựng notebook và trang web nằm trong scratchpad của session cũ và có thể đã mất. Notebook hiện tại là nguồn chính.

## 7. Trang đã xuất bản (claude.ai artifacts)

Muốn nhờ Claude đọc hoặc sửa hai trang này thì dán link kèm yêu cầu. Cách đọc, các giới hạn về kích
thước và quy tắc publish lại nằm ở `SESSION_LOG.md` mục "Hai trang đã xuất bản".

- Báo cáo đánh giá pipeline (bản 3, 2026-11-04): https://claude.ai/artifact/SDqYDmZePgebmz9MmQULtT — chia sẻ "anyone with the link", **người có link thấy bản mới ngay**. Đã khớp cấu trúc 2 notebook: cột "Ở đâu" và mọi con trỏ §x.y vào code giờ ghi theo số mục của notebook. Kết luận và số liệu không đổi (vẫn là của lần chạy 1 seed).
- Hai thí nghiệm 03 và 04 (trang mới, 2026-11-04): https://claude.ai/artifact/AAPr3JDeJhbhqZsXCDATzt — riêng tư. Giải thích bằng tiếng Việt vì sao cần 03 và 04, đo gì, kết quả nói gì. Viết cho người cần hiểu để trình bày, không cần đọc code. Nhúng 3 hình base64.
- Giải thích code từng notebook: https://claude.ai/artifact/S29cQ3jSfvgLmXRypbDw1v — **anyone with the link** (người có link thấy bản mới ngay). **Bản 6 (2026-11-04, phiên thứ ba)**: bỏ giải thích các cell vẽ hình (thay bằng một dòng), viết lại 01 mục 6–10 và 02 mục 2–11 cho thành viên đọc và giải thích lại được: 3 sơ đồ SVG (cửa sổ trượt, ranh giới split, vòng lặp epoch), khung "Giải thích lại trong 30 giây" và "Câu hỏi hay gặp" ở mỗi phần. 24 khối code, đều khớp notebook. Chi tiết: `SESSION_LOG.md`. Ghi chú bản 5 bên dưới: bản 3 khớp notebook 02 sau khi tách 3 mô hình: mục 2 quy trình chung, mục 3.1/3.2/3.3 mỗi mô hình một khối code kèm chú thích (số tham số, cổng của GRU/LSTM), mục 10 chỉ còn hình sai số. Đã kiểm tra bằng script: **cả 32 khối code trên trang trùng đúng với cell của 2 notebook**. Cách dựng lại: xem `SESSION_LOG.md`.

## 8. Nhật ký

- **2026-11-03:** Review toàn bộ project. Bỏ phần trùng lặp giữa 01/02/03. Đánh giá pipeline (trang báo cáo). Đối chiếu với đề PDF và sửa lại kết luận: target mức giá là giới hạn của đề, không phải lỗi. Refactor 01–03 theo thứ tự PDF, đánh số mục 1, 2, 3. Thêm 5 seed, baseline trên val, bỏ `shuffle=False`, 02 lưu sẵn sequences cho 4 độ dài window. Tạo trang giải thích code. Chưa chạy lại 02/03, chưa commit.
- **2026-11-04:** Theo yêu cầu của user: bỏ đoạn xử lý missing value khỏi notebook tiền xử lý (dữ liệu đã xác nhận sạch), rồi **gộp 01 + 02 thành `01_data.ipynb`** và đổi tên `03_model_training.ipynb` → `02_model_training.ipynb`. Gộp luôn 2 folder results thành `results/01_data/`. Viết lại markdown của 01: bảng mục lục mục → requirement ở đầu file, mỗi mục có phần mô tả, mục nhiều cell thì chia `### x.y` để thấy ranh giới cell. Hợp nhất 2 bảng summary split thành 1 (có cả USD và scaled). Bỏ phần nhắc requirement §x.y trong markdown của từng mục (đã có bảng tóm tắt ở đầu file). Dựng lại cả hai trang đã xuất bản theo cấu trúc 2 notebook. Cập nhật README + SESSION_NOTES. **Đã chạy lại 01, không lỗi**; 02 chưa chạy. Chưa commit. Sau đó (cùng ngày): viết lại 02 để mỗi mô hình một cell riêng, gộp bảng overfitting vào bảng persistence, bỏ bảng trùng với 01, bỏ hình gộp; chạy thử bản rút gọn OK. Dựng lại trang "Đọc hiểu code" (bản 3) theo notebook 02 mới. **User chạy 02 (5 seed)**; đã kiểm tra output, cập nhật README + Discussion 02 mục 10 + cả hai trang theo số mới (bản 4). Hỏi đáp: chồng thêm layer gần như chắc chắn không giúp vì nút thắt là ngoại suy, không phải capacity (chưa chạy thí nghiệm kiểm chứng).
- **2026-11-04 (phiên thứ ba):** Viết lại trang "Đọc hiểu code" thành bản 6 theo yêu cầu user: bỏ phần giải thích cell vẽ hình, làm lại phần hệ thống của 01 (chuẩn hóa, cắt window, leakage, split) và toàn bộ 02 từ training procedure trở đi, thêm đoạn giải thích lại + Q&A. Không sửa notebook. Chưa commit.
