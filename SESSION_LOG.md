# Session log — chi tiết từng session

`SESSION_NOTES.md` là file **trạng thái**: đọc nó trước để biết project đang ở đâu.
File này là **mục lục các phiên làm việc**: mỗi phiên một mục tóm tắt, không sửa mục của phiên cũ.

**Quy ước từ 2026-11-04 trở đi:** phiên nào dài thì có **file riêng** `SESSION_<ngày>_<chủ đề>.md`, còn ở
đây chỉ để tóm tắt và con trỏ. Phiên ngắn thì ghi thẳng vào đây như cũ.

### Các file nhật ký riêng

| File | Nội dung |
|---|---|
| [`SESSION_2026-11-04_experiments.md`](SESSION_2026-11-04_experiments.md) | Viết lại 02, chạy 02 (5 seed), notebook 03 + 04, cả ba trang xuất bản. **Chứa: đề xuất cải tiến chưa làm (ensemble, `MAX_EPOCHS`), danh sách nguồn tham khảo kèm URL, và 10 bẫy kỹ thuật.** |

Cập nhật lần cuối: 2026-11-04

---

## Ba trang đã xuất bản

Ba trang này là tài liệu chính của project, ngoài notebook. Mỗi trang có **một đối tượng đọc khác nhau** —
đừng gộp nội dung giữa chúng.

| Trang | Link | Chia sẻ | Viết cho ai | Nội dung |
|---|---|---|---|---|
| **Thẩm định pipeline** (bản 4) | https://claude.ai/artifact/SDqYDmZePgebmz9MmQULtT | anyone with the link | Người chấm / review kỹ thuật | Đối chiếu với đề, đánh giá từng giai đoạn, 10 lỗi kinh điển, **10** phát hiện cụ thể, việc còn lại. **Chưa nhắc tới notebook 03 và 04** |
| **Đọc hiểu code** (bản 6) | https://claude.ai/artifact/S29cQ3jSfvgLmXRypbDw1v | anyone with the link (lần `read` 2026-11-04 báo vậy, trước ghi nhầm là riêng tư) | Thành viên nhóm cần đọc hiểu code **và giải thích lại được** | Giải thích từng cell của 01 và 02. Bản 6: bỏ giải thích các cell vẽ hình, viết lại 01 mục 6–10 và 02 mục 2–11, mỗi phần có khung "Giải thích lại trong 30 giây" + "Câu hỏi hay gặp" |
| **Hai thí nghiệm 03 + 04** | https://claude.ai/artifact/AAPr3JDeJhbhqZsXCDATzt | riêng tư | Người cần hiểu để **trình bày**, không đọc code | Vì sao cần 03 và 04, đo gì, kết quả nói gì. Tiếng Việt, nhiều ẩn dụ, 3 hình nhúng |

### Cách nhờ Claude đọc ba trang này ở session sau

Chỉ cần dán link và nói muốn gì, ví dụ:

> Đọc trang https://claude.ai/artifact/SDqYDmZePgebmz9MmQULtT rồi cho tôi biết phát hiện số 3 nói gì.

Lưu ý khi làm việc với hai trang này:

- Claude đọc chúng bằng **Artifact tool, action `read`**, không phải WebFetch hay curl. Link
  `claude.ai/artifact/...` không fetch được như trang web thường.
- Trang "Đọc hiểu code" nặng (kèm bundle highlight.js khi xuất bản) và trang "Thẩm định" có một ảnh
  base64 lớn, nên **không đọc cả file một lần**. Cách đã dùng: `read` để lấy phần đầu, rồi lọc bỏ
  các `data:image/...;base64,...` khỏi bản đã lưu trong `tool-results/` để xem phần còn lại.
- **Muốn sửa thì phải `read` trước khi publish**, nếu không bản publish sẽ bị từ chối.
- Publish lại bằng cùng `url` thì giữ nguyên link. Publish không kèm `url` sẽ tạo trang mới.
- Trang Thẩm định đang chia sẻ public-link, nên **người có link thấy bản mới ngay sau khi publish**.

---

## Nhật ký chi tiết

### 2026-11-04 (phiên thứ ba) — Viết lại trang "Đọc hiểu code" (bản 6)

User yêu cầu: bỏ phần giải thích liên quan tới vẽ hình; làm lại các phần hệ thống (normalization, check
leakage, split) và đặc biệt mọi phần từ training procedure trở đi, dễ hiểu và chi tiết để các thành viên
khác đọc, hiểu và **giải thích lại được**.

Đã làm (không sửa notebook, chỉ sửa trang):

- **Cell vẽ hình** (01: 5.1, 5.2, 9.3, 9.4; 02: mục 5, 6, hình mục 9, hình mục 10) thay bằng một dòng gạch đứt
  "Chỉ vẽ hình, trang bỏ qua". Cell 5.3 (tương quan) giữ code nhưng chỉ giải thích phần `corr`. Bỏ các chú thích
  `plt.*` ở cell setup. 32 → **24 khối code**, script assert cả 24 trùng nguyên văn với cell notebook.
- **01 mục 6–10 viết lại:** vì sao scale (tanh bão hòa), công thức với số thật (min 50,06, max 835,74, bảng
  USD → scaled), fit vs transform (ẩn dụ cây thước), vì sao fit toàn bộ là rò rỉ (train bị nén vào [0; 0,267]),
  vì sao hai scaler; sơ đồ SVG cửa sổ trượt; bảng 6 assert → điều kiện → lỗi nó bắt; vì sao chia theo thời
  gian, vai trò train/val/test, sơ đồ SVG ranh giới train|val với mẫu val đầu tiên mượn input; bảng tóm tắt
  627/665 và 665/665.
- **02 mục 2 viết lại hoàn toàn:** sơ đồ SVG vòng lặp một epoch (96 batch = 3.071/32), MSE scaled ↔ USD
  (`√val_loss × 785,68`), Adam + learning rate (dẫn số mục 9), early stopping với dải epoch thật của SimpleRNN
  seed 42 (best 65, dừng 75), 12/15 lượt `epochs run = best + 10`, 3 lượt LSTM chạm trần 100; shuffle không rò
  rỉ; 5 seed và bảng `runs` vs `best`.
- **02 mục 3:** giữ phần cơ chế RNN của bản 5, thêm khung nói lại; 3.1 thêm chú thích *Total params 12.869* =
  4.289 trainable + 8.580 của Adam (2 số/trọng số + 2).
- **02 mục 4, 7, 8, 9, 10 viết lại:** bảng 4 metric (công thức, đơn vị, cách đọc, cảnh báo R²), persistence,
  đọc một dòng kết quả; bảng ratio và 3 bước đọc; cách đọc ±; bảng HP study có ý nghĩa từng yếu tố; mục 10 là
  chuỗi nguyên nhân ngoại suy 6 bước + overfit/underfit + các giới hạn, có một dòng trỏ sang notebook 04.
- Thêm CSS: `.skip`, `.say`, `.qa` (dùng `<details>`), `.steps`, `.tbl`, `.epochs`, `.chain`, token `--c-val`,
  `--c-test` (cả light/dark). TOC thêm 2 link: `#sys` (01 mục 6) và `#eval` (02 mục 4).

Script: `build_v6.py` (scratchpad, sẽ mất). Cách làm: đọc bản `read` đã lưu, cắt khung publish ở đầu và bundle
hljs ở cuối, tách các `<section class="nb-sec">` theo tiêu đề `h3`, giữ nguyên mục 1–4 của 01 và mục 1, 3, 3.1–3.3,
11 của 02, dựng mới phần còn lại; code luôn lấy thẳng từ `.ipynb` rồi assert khớp.

### 2026-11-04 (phiên thứ hai) — Viết lại 02, chạy 02, notebook 03 và 04, trang thứ ba

> **Nhật ký đầy đủ của phiên này nằm ở file riêng: [`SESSION_2026-11-04_experiments.md`](SESSION_2026-11-04_experiments.md).**
> Từ phiên này trở đi, mỗi phiên có một file `SESSION_<ngày>_<chủ đề>.md` riêng; mục ở đây chỉ là tóm tắt
> và con trỏ, để file này không phình to mãi.

Tóm tắt những gì phiên đó làm:

| | Trước | Sau |
|---|---|---|
| Notebook | 2 | **4** (thêm `03_direction_metrics`, `04_skip_connection`) |
| Notebook 02 | 3 mô hình gộp trong 1 hàm | **mỗi mô hình một cell riêng** |
| Lần chạy 02 | Chưa chạy | **Đã chạy 5 seed**, mọi tài liệu cập nhật theo |
| Trang xuất bản | 2 | **3** |

Ba kết quả đáng nhớ nhất:

1. **Tách được hai vấn đề** mà 4 metric bắt buộc gộp làm một: *(a)* target ra ngoài miền train — chỉ ảnh
   hưởng test, **sửa được**; *(b)* chuỗi gần bước ngẫu nhiên — ảnh hưởng cả val lẫn test, **không sửa được**.
2. **Skip connection xác nhận chẩn đoán**: đổi một dòng, RMSE test tốt hơn **2,7–5,3 lần**, khoảng cách tới
   baseline từ 2,90–5,77× xuống 1,05–1,10×. Capacity chưa bao giờ là nút thắt.
3. **Bẫy lớn nhất**: con số validation của notebook 04 trông "có ý nghĩa thống kê" (p 0,006–0,023) nhưng
   **không được tin**, vì chính validation đã chọn ra các model đó. Test mới là ước lượng sạch.

### 2026-11-04 — Gộp phần dữ liệu thành một notebook

Session này xuất phát từ nhận xét của user, không phải từ một kế hoạch có sẵn. Thứ tự diễn biến:

#### 1. Hỏi đáp về input / output

User hỏi feature nào là input, cái gì là output, và liệu thêm feature có tăng độ chính xác không.

- Input: `FEATURES = ["Close"]`, mỗi mẫu là Close của 30 ngày trước (`t-30` → `t-1`), shape `(30, 1)`.
- Output: `TARGET = "Close"`, giá đóng cửa **ngày kế tiếp** (ngày `t`).
- Có một dòng đang comment để chạy đa biến `["Open","High","Low","Close","Volume"]`, chưa dùng.

**Kết luận đã trả lời user:** thêm feature khó cải thiện, vì nút thắt không phải thiếu thông tin.
Open/High/Low tương quan với Close > 0,999 nên gần như không thêm thông tin; Volume tương quan
−0,45 và khác thang đo hoàn toàn. Quan trọng hơn: vấn đề là **target nằm ngoài miền train**
(665/665 ngày test vượt 835,74 USD), nên thêm feature thì chính các feature đó cũng bị ngoại suy.
Cách thực sự có thể cải thiện là đổi target sang log-return — nhưng đề bắt buộc target là giá, nên
chỉ làm được như thí nghiệm phụ. Nếu cần thêm nội dung cho §8 thì có thể bật dòng multivariate,
chạy với LSTM, và báo cáo rằng **không khá hơn** — một kết quả âm có dẫn chứng vẫn là phân tích tốt.

#### 2. User phát hiện 4 cell đầu của 01 và 02 trùng nhau

Giải thích đã đưa ra: notebook không chia sẻ state nên 02 buộc phải tự `read_csv` + `sort_values`;
và nội dung thực chất khác nhau — 01 **chẩn đoán** (có `describe()`, kiểm tra monotonic, đếm ngày
trùng), 02 **thi hành** (có `ffill()` + `dropna()` cho §4.3).

**User không đồng ý giữ:** dữ liệu đã xác nhận sạch thì không cần viết code cho nó, và cell nào đã
có ở 01 thì không viết lại ở 02 trừ khi cần thiết.

→ Đã xóa mục "Handle missing values" khỏi 02, chuyển `n = len(df)` sang cell sort, đánh số lại các
mục phía sau, và thêm markdown nói rõ §4.3 không cần imputation.

#### 3. User yêu cầu gộp 01 và 02 thành một file

Lý do user đưa ra: cả hai đều liên quan đến data, gộp lại dễ quan sát hơn.

Đã hỏi user một câu về cách đánh số và **user chọn: `01_data` + `02_model_training`** (đánh số liền
mạch). Các thay đổi:

| Trước | Sau |
|---|---|
| `code/01_data_explore.ipynb` + `code/02_preprocessing.ipynb` | `code/01_data.ipynb` |
| `code/03_model_training.ipynb` | `code/02_model_training.ipynb` |
| `results/01_data_explore/` + `results/02_preprocessing/` | `results/01_data/` |
| `results/03_model_training/` | `results/02_model_training/` |

Tham chiếu bên trong `02_model_training.ipynb` đã sửa: "Run notebook 02 first" → "notebook 01",
`RESULTS_DIR`, "written by notebook 02" → "notebook 01", và tiêu đề mục 1.

#### 4. Cấu trúc notebook 01 sau khi gộp

10 mục, 39 cell (17 code / 22 markdown). User yêu cầu markdown phải đủ để **người khác đọc vẫn biết
ranh giới các cell code và biết cell đó làm gì**, nên:

- Đầu file có **bảng mục lục** map mục → requirement → việc làm gì, cộng dòng ghi rõ đọc gì / ghi gì.
- Mỗi mục `##` có đoạn mô tả, không chỉ tiêu đề trơ.
- Mục nào nhiều cell code thì chia tiếp `### x.y`, nên **mỗi cell code đều có markdown riêng ngay
  phía trên**. Ví dụ mục 5: 5.1 giá và volume, 5.2 phân phối, 5.3 tương quan, 5.4 quyết định.
  Mục 9: 9.1 bảng tổng hợp, 9.2 assert, 9.3 hình theo USD, 9.4 hình sau scale.

Map mục → đề: 2–4 = §4.1–4.3 · 5 = §4.4 · 6 = §4.5 · 7 = §4.6 · 8 = §4.7 · 9 = §5 · 10 = ghi file.

Hai thay đổi nội dung đáng nhớ:

- **Hợp nhất 2 bảng summary split thành 1.** Trước đây 01 có bảng theo USD (mean, max, days above
  train max), 02 có bảng theo scaled (sequences, scaled max). Giờ một bảng có cả hai thang.
- **`train_end` tính ở mục 6** (normalize) vì scaler cần biên train trước; `val_end` ở mục 9.
  Markdown đã giải thích vì sao thứ tự này là bắt buộc, không phải lỗi sắp xếp.

#### 5. Bỏ nhắc requirement trong markdown từng mục

User: *"các markdown phía trên mỗi cell code không cần nhắc thêm requirement, ta có thể xem ở bảng
tóm tắt được rồi"*. Đã bỏ 12 chỗ: 8 tiêu đề mục (`— requirement §4.x`) và 4 chỗ lẫn trong phần mô tả.

**Giữ lại:** bảng tóm tắt ở đầu file (chỗ duy nhất còn map mục → requirement), và **các comment
trong code** (`# requirement §4.6: 30 trading days in...`) vì user chỉ nói về markdown.
Nếu session sau muốn bỏ luôn comment thì hỏi user trước.

#### 6. Chạy lại notebook 01 — đã verify

Chạy không lỗi, mọi assert pass, 6 hình nhúng trong notebook. Số liệu khớp đúng với kết luận cũ:

```
Shape: (4431, 7)           Rows: 4431      Chronological: True    Duplicate dates: 0
Missing: 0 mọi cột · Non-positive prices: 0 · High < Low: 0 · gap lớn nhất 5 ngày · Close == Adj Close: True
Correlation với Close: Open 0,9998 · High 0,9999 · Low 0,9999 · Volume −0,4543
scaled Close — train: 0,000 → 1,000 | sau train: 0,945 → 3,751
X=(4401, 30, 1)  y=(4401,)   → all checks passed, split checks passed

split  rows  seq   start       end         Close mean  Close max  days>train max  scaled max
train  3101  3071  2004-08-19  2016-12-09  350,70      835,74     0               1,000
val     665   665  2016-12-12  2019-08-05  1052,03     1296,20    627             1,586
test    665   665  2019-08-06  2022-03-24  1930,84     2996,77    665             3,751
```

Notebook 02 **chưa chạy** (khoảng 40 phút, user nói sẽ tự test).

#### 7. Cập nhật README

Cây thư mục, hướng dẫn chạy ("Run `01_data.ipynb`, then `02_model_training.ipynb`"), 3 link ảnh
`results/03_model_training` → `results/02_model_training`, và chỗ trích `FEATURES`.

#### 8. Dựng lại hai trang đã xuất bản

- **Đọc hiểu code** (bản 2): gộp phần 01 + 02 cũ thành một article, bỏ mục missing value, thêm chú
  thích cho các dòng mới (`n = len(df)`, `bounds`, `df[TARGET][a:b]`), đổi article 03 → 02. Phần
  notebook 01 trên trang **sinh tự động từ file .ipynb** nên code trên trang không thể lệch khỏi
  code thật.
- **Thẩm định pipeline** (bản 3): 49 thay đổi, mỗi chỗ khớp đúng một lần. Cột "Ở đâu" và mọi con trỏ
  `§x.y` vào code giờ ghi theo số mục notebook (ví dụ `02 §4.5` → `01 mục 6`, `03 §8.6` → `02 mục 10`).
  **Sửa một chỗ đã thành sai:** dòng "Điền giá trị thiếu dùng thông tin tương lai" trước ghi "Chỉ dùng
  `ffill` (nhìn lùi)" — giờ ghi rằng file không có NaN và đoạn `ffill`/`dropna` đã bị bỏ vì là dead code.
  **Kết luận, số liệu, verdict giữ nguyên** vì vẫn là của lần chạy 1 seed.

---

## Bẫy kỹ thuật đã gặp (đọc trước khi lặp lại)

> Phiên 2026-11-04 (phiên thứ hai) gặp thêm **10 bẫy nữa** — tên lớp Keras tự đánh số, slicing
> KerasTensor, heredoc hỏng script Python, persistence không dùng làm baseline cho directional accuracy,
> khung HTML do publish tự chèn… Tất cả nằm ở mục 9 của
> [`SESSION_2026-11-04_experiments.md`](SESSION_2026-11-04_experiments.md).

1. **In tiếng Việt qua Bash tool bị `UnicodeEncodeError`.** Python trên máy này dùng codec `cp1252`
   cho stdout, nên `print()` chuỗi có dấu sẽ crash. Cách xử lý: đặt `PYTHONIOENCODING=utf-8` trước
   lệnh, hoặc ghi ra file UTF-8 rồi đọc file bằng Read tool. Đừng đoán, nó sẽ fail ngay.

2. **`MPLBACKEND=Agg` làm mất hình nhúng trong notebook.** Chạy nbconvert với biến này thì
   `plt.show()` không tạo output, notebook ra 43 KB và không có hình nào. Phải để kernel dùng backend
   inline mặc định; khi đó notebook ra ~317 KB và có 6 ảnh PNG nhúng.

3. **Cell thiếu field `id`** gây `MissingIDFieldWarning` khi nbconvert validate. Đã thêm `id` cho mọi
   cell của `01_data.ipynb`. Nếu dựng notebook bằng script thì nhớ set `id`.

4. **Không chạy `python -I`.** Pandas / scikit-learn / TensorFlow nằm ở user-site, cờ `-I` bỏ qua
   user-site và gây lỗi giả "không có pandas". Xem thêm `SESSION_NOTES.md` mục 6.

5. **Artifact phải `read` trước khi publish lại.** Publish vào một URL mà session hiện tại chưa đọc
   sẽ bị từ chối.

---

## Script đã dùng (nằm trong scratchpad, có thể đã mất)

Scratchpad là theo session nên các file này **không còn ở session sau**. Ghi lại để biết cách làm lại:

| Script | Việc |
|---|---|
| `trim02.py` | Xóa mục missing value khỏi notebook tiền xử lý, chuyển `n = len(df)`, đánh số lại mục |
| `build_merged.py` | Dựng `01_data.ipynb` từ đầu bằng JSON, gồm toàn bộ markdown |
| `build_page.py` | Dựng lại trang "Đọc hiểu code"; đọc code trực tiếp từ `.ipynb`, tái dùng CSS và article 03 từ bản đã xuất bản |
| `fix_report.py` | 49 phép thay thế chuỗi chính xác trên trang "Thẩm định", có assert mỗi chuỗi khớp đúng 1 lần |

Script của phiên 2026-11-04 thứ hai (14 script nữa, gồm dựng notebook 03 và 04, dựng trang thứ ba, sửa hai
trang cũ) nằm ở mục 10 của [`SESSION_2026-11-04_experiments.md`](SESSION_2026-11-04_experiments.md).

Cách làm lại khi cần: `read` artifact → bản đầy đủ được lưu vào `tool-results/` → viết script thay
thế chuỗi trên bản đó → publish lại bằng cùng `url`. Với notebook thì sửa trực tiếp file `.ipynb`.
