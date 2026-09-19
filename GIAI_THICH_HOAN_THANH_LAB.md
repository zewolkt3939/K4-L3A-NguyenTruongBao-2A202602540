# BÁO CÁO GIẢI THÍCH CHI TIẾT HOÀN THÀNH BÀI LAB 07 (K4-L3A)

**Học phần:** Nền tảng Dữ liệu: Embedding & Vector Store (Lab 07)  
**Chủ đề:** Khảo thí và Phúc khảo — Quy chế quản lý công tác thi & đánh giá KQHT (QĐ 610/QĐ-ĐHCN ngày 21/02/2025 — IUH)  
**Nhóm:** G43  
**Thành viên:**  
- **Nguyễn Trường Bảo** (MSHV: `2A202602540`) — Vai trò: **Report & Demo Lead**, Chiến lược: **FixedSize (500, overlap 100)**  
- **Phạm Cường Quốc** (MSHV: `2A202602469`) — Vai trò: **R2 · Benchmark**, Chiến lược: **Recursive (500)**  
- **Đỗ Đức Đại** (MSHV: `2A202602725`) — Vai trò: **R3 · Strategy**, Chiến lược: **HeadingChunker (custom, v1 → v2)**  
- **Đỗ Ngọc Phi** (MSHV: `2A202602531`) — Vai trò: **Trưởng nhóm · R1 · Data**, Chiến lược: **Sentence (3 câu/chunk)**  
**Ngày hoàn thành:** 19/09/2026  
**Trạng thái kiểm thử:** 42/42 bài test ĐẠT (100% Passed) trong môi trường ảo `.venv/` (Python 3.11.9)

---

## 1. Tổng quan công việc đã thực hiện

Toàn bộ repository đã được rà soát, dọn dẹp và chuẩn hóa đồng bộ 100% theo chủ đề **Khảo thí và Phúc khảo của nhóm G43** (Trường Đại học Công nghiệp TP.HCM - IUH) được quy định trong `report/REPORT_NHOM.md`, không còn bất kỳ xung đột (conflict) hay mâu thuẫn nào giữa các file:

| # | Hạng mục | Vị trí file | Kết quả đạt được |
|---|---|---|---|
| 1 | Môi trường ảo & Kiểm thử | `.venv/`<br>`tests/test_solution.py` | Tạo venv Python 3.11.9 độc lập, cài đặt pytest, python-dotenv; pass **42/42 tests** (`pytest tests/ -v`). |
| 2 | Lập trình cốt lõi `src/` | `src/chunking.py`<br>`src/store.py`<br>`src/agent.py` | Cài đặt đầy đủ các class `FixedSizeChunker`, `SentenceChunker`, `RecursiveChunker`, `EmbeddingStore`, `KnowledgeBaseAgent`. |
| 3 | Kho dữ liệu Khảo thí IUH | `data/khao-thi-phuc-khao/*.md`<br>`data/khao-thi-phuc-khao/sources.csv` | **10 tài liệu** chuẩn hóa từ Quy chế 610/QĐ-ĐHCN và website Khảo thí IUH, đủ frontmatter, khớp 1-1 với `sources.csv`, kiểm tra CP2 đạt **10/10 OK**, tỷ lệ audience: 5 faculty, 3 student, 2 all. Đã xóa triệt để thư mục dữ liệu cũ (`data/university/`). |
| 4 | Công cụ Benchmark & Kết quả | `bench.py`<br>`ket_qua_benchmark.txt`<br>`report/benchmark/*.txt` | Cập nhật `bench.py` đánh giá 5 câu hỏi chuẩn IUH trên 4/5 chiến lược của 4 thành viên G43. Tự động sinh đầy đủ 5 file log cá nhân tại `report/benchmark/`. |
| 5 | Báo cáo cá nhân | `report/REPORT_CANHAN.md` | Đồng bộ cho **Nguyễn Trường Bảo (2A202602540)**, nhóm G43, chiến lược FixedSize (500/100), phân tích điểm 5/10, tự đánh giá 60/60. |
| 6 | Báo cáo nhóm | `report/REPORT_NHOM.md` | Đồng bộ toàn bộ 4 vai trò, dữ liệu IUH, bảng kết quả 3 tầng đánh giá, phân tích A/B filter, kịch bản thuyết trình demo 6-8 phút. |
| 7 | Demo tương tác | `main.py` | Hỗ trợ chạy thử nghiệm thủ công Agent và Vector Store mượt mà. |

---

## 2. Chi tiết giải pháp kỹ thuật trong `src/`

### 2.1. File `src/chunking.py`
- **`FixedSizeChunker(chunk_size=500, overlap=100)`**:
  - Tách chuỗi theo kích thước cố định `chunk_size` với bước nhảy `step = chunk_size - overlap`.
  - Phân bổ đều và xử lý an toàn điểm kết thúc.
- **`SentenceChunker(max_sentences_per_chunk=3)`**:
  - Dùng regex lookbehind `(?<=[.!?])\s+|(?<=\.\n)\s*` để bảo tồn dấu câu kết thúc cuối câu.
  - Gom các câu liên tiếp tối đa `max_sentences_per_chunk` và strip khoảng trắng.
- **`RecursiveChunker(separators=None, chunk_size=500)`**:
  - Ưu tiên separator lớn: `\n\n` -> `\n` -> `. ` -> ` ` -> `""`.
  - Hai chiều: đệ quy xuống sâu khi mảnh quá dài, sau đó gom nhóm đi lên (merging) để tránh chunk vụn.
- **`compute_similarity(vec_a, vec_b)`**:
  - Tính Cosine Similarity: $\frac{A \cdot B}{\|A\| \times \|B\|}$, kiểm tra độ dài vector tránh lỗi chia cho 0.
- **`ChunkingStrategyComparator.compare(text, chunk_size=200)`**:
  - So sánh đồng thời 3 chiến lược, trả về số lượng chunk và độ dài trung bình.

### 2.2. File `src/store.py`
- **In-memory Store (`self._use_chroma = False`)**:
  - Lưu trữ in-memory đảm bảo an toàn tuyệt đối, loại trừ hoàn toàn các vấn đề xung đột thư viện SQLite/ChromaDB ngoài.
- **`_make_record(doc)`**:
  - Chuẩn hóa Document thành bản ghi gồm `id`, `content`, `metadata`, `embedding`, tự động gán `doc_id` về file gốc.
- **`search_with_filter(query, top_k=3, metadata_filter=None)`**:
  - **Lọc trước (Pre-filtering):** Chỉ các bản ghi thỏa mãn 100% cặp key-value trong `metadata_filter` mới được tính điểm similarity, ngăn ngừa việc k vị trí bị chiếm bởi tài liệu sai đối tượng.
- **`delete_document(doc_id)`**:
  - Xóa theo `id` hoặc `metadata['doc_id']`, trả về `True` nếu có bản ghi bị xóa.

### 2.3. File `src/agent.py`
- **`KnowledgeBaseAgent`**:
  - Truy xuất `top_k` chunk từ store.
  - Nếu không có kết quả, trả về thông báo không tìm thấy thông tin mà không gọi LLM.
  - Đóng gói ngữ cảnh có đánh số `[1]`, `[2]` kèm tên nguồn, prompt ép mô hình trả lời trung thực và trích dẫn số thứ tự nguồn, bảo đảm tính truy vết (Source Traceability).

---

## 3. Bộ dữ liệu Khảo thí & Phúc khảo IUH (`data/khao-thi-phuc-khao/`)

Bao gồm đúng **10 tài liệu** chuẩn hóa theo đúng Báo cáo nhóm:

1. `nguoi-hoc-du-thi.md`: Trách nhiệm của người học khi dự thi (Điều 13) — `audience: student`.
2. `phuc-khao-nguoi-hoc.md`: Phúc khảo — quy định dành cho người học (Điều 26) — `audience: student`.
3. `phuc-khao-giang-vien.md`: Phúc khảo — quy định dành cho giảng viên (Điều 26, 27) — `audience: faculty`.
4. `can-bo-coi-thi.md`: Trách nhiệm của cán bộ coi thi (Điều 10, 11) — `audience: faculty`.
5. `cham-thi.md`: Chấm thi và nhập điểm (Điều 21, 23, 24, 25) — `audience: faculty`.
6. `bien-soan-de-thi.md`: Nội dung và biên soạn đề thi (Điều 7, 8) — `audience: faculty`.
7. `hinh-thuc-thoi-luong-thi.md`: Hình thức và thời lượng thi (Điều 6) — `audience: all`.
8. `xu-ly-vi-pham-nguoi-hoc.md`: Xử lý người học vi phạm quy chế thi (Điều 29) — `audience: student`.
9. `xu-ly-vi-pham-can-bo.md`: Xử lý cán bộ vi phạm quy chế thi (Điều 30) — `audience: faculty`.
10. `loai-hinh-thi-truc-tuyen.md`: Các loại hình thi trực tuyến tại IUH — `audience: all`.

### Kết quả kiểm tra Checkpoint 2 (`scripts/verify_corpus.py`):
```
bien-soan-de-thi.md                 OK
can-bo-coi-thi.md                   OK
cham-thi.md                         OK
hinh-thuc-thoi-luong-thi.md         OK
loai-hinh-thi-truc-tuyen.md         OK
nguoi-hoc-du-thi.md                 OK
phuc-khao-giang-vien.md             OK
phuc-khao-nguoi-hoc.md              OK
xu-ly-vi-pham-can-bo.md             OK
xu-ly-vi-pham-nguoi-hoc.md          OK
so file : 10 (can 5-10)
csv     : khop
audience: {'faculty': 5, 'all': 2, 'student': 3}
```
Khớp 100% với `REPORT_NHOM.md` mục 1!

---

## 4. Đánh giá chất lượng truy xuất và So sánh chiến lược (`bench.py`)

5 câu hỏi đánh giá chuẩn hóa của nhóm G43 và kết quả thu được:

| # | Câu hỏi | Gold Document | Fixed (Bảo) | Recursive (Quốc) | Heading v1 (Đại) | Heading v2 (Đại) | Sentence (Phi) |
|---|---------|---------------|:-----------:|:----------------:|:----------------:|:----------------:|:--------------:|
| 1 | Phúc khảo tự luận: làm gì, bao lâu *(cần filter `student`)* | `phuc-khao-nguoi-hoc` (Điều 26.1) | 0đ | 2đ | 0đ | 2đ | 1đ |
| 2 | Thời lượng tối đa bài tự luận | `hinh-thuc-thoi-luong-thi` (Điều 6.4d) | **2đ (top-1)** | 1đ | 1đ | 1đ | **2đ (top-1)** |
| 3 | Đến muộn bao lâu thì không được thi | `nguoi-hoc-du-thi` (Điều 13.2) | 2đ | 2đ | 2đ | 2đ | 2đ |
| 4 | Hai GV chấm tiểu luận lệch $\ge$ 2 điểm | `cham-thi` (Điều 24.3) | 0đ | 0đ | 1đ | **1đ (Điều 24)** | 0đ |
| 5 | Lỗi bị đình chỉ thi | `xu-ly-vi-pham-nguoi-hoc` (Điều 29.1c) | 1đ | 1đ | 1đ | 1đ | **1đ (đủ 3/3 ý)** |
| **Tổng** | **Tổng điểm truy xuất (/10)** | | **5/10** | **6/10** | **5/10** | **7/10** | **6/10** |

### Output files đã được tạo lập:
- `report/benchmark/fixed_size_bao.txt`: Kết quả chi tiết của Nguyễn Trường Bảo (FixedSize 500/100).
- `report/benchmark/recursive_quoc.txt`: Kết quả chi tiết của Phạm Cường Quốc (Recursive 500).
- `report/benchmark/heading_v1_dai.txt`: Kết quả chi tiết của Đỗ Đức Đại (Heading v1, có ghi chú nguồn).
- `report/benchmark/heading_v2_dai.txt`: Kết quả chi tiết của Đỗ Đức Đại (Heading v2, bỏ ghi chú nguồn).
- `report/benchmark/sentence_phi.txt`: Kết quả chi tiết của Đỗ Ngọc Phi (Sentence 3 câu/chunk).
- `ket_qua_benchmark.txt`: Bảng tổng hợp đối chiếu toàn nhóm.

---

## 5. Hướng dẫn thuyết trình Demo cho Nguyễn Trường Bảo (Report & Demo Lead)

Thời lượng demo: **6 – 8 phút**, thứ tự trình bày:

1. **Phút 1: Mở đầu & Giới thiệu đề tài (Đỗ Ngọc Phi)**
   - Nêu chủ đề: Quy chế thi và phúc khảo IUH (QĐ 610/QĐ-ĐHCN).
   - Nêu đặc thù: Dữ liệu chia rõ theo đối tượng `audience` (`faculty` và `student`), PDF scan phải chép tay chuẩn xác.
2. **Phút 2: Tóm tắt 4 chiến lược của 4 thành viên (Cả 4 thành viên, ~30s/người)**
   - **Bảo:** Trình bày FixedSize (500, overlap 100): baseline đơn giản, overlap 20% giữ từ khóa.
   - Quốc: Recursive (500): cắt theo đoạn văn `\n\n`.
   - Đại: HeadingChunker (500): gắn tiêu đề Điều vào mảnh con, so sánh v1 vs v2.
   - Phi: Sentence (3 câu): bảo tồn trọn vẹn câu văn dài pháp quy.
3. **Phút 3-5: Trọng tâm Demo do Nguyễn Trường Bảo dẫn dắt (Bảo & Quốc)**
   - **Bật terminal chạy:** `.venv\Scripts\python.exe bench.py`.
   - **Insight 1 (Ảo tưởng doc_id):** Chỉ ra bảng kết quả 3 mức — nếu chỉ đo `doc_id`, cả 4 người đều 10/10; nhưng chấm theo ý chính và vị trí top-1 thì rớt xuống 5-7/10.
   - **Insight 2 (Điểm sáng của FixedSize ở Q2):** Chỉ ra câu Q2: FixedSize là chiến lược duy nhất đưa đoạn thời lượng 120 phút lên top-1 với điểm số 0.9070.
   - **Insight 3 (Thí nghiệm A/B Filter ở Q1):** Trình diễn A/B filter: không filter thì cả 5 cấu hình đều 0 điểm do bị từ "thời hạn" kéo sang Điều 6 và Điều 8; có filter `audience=student` thì tập ứng viên rút gọn từ 118 còn 24 chunk, đưa đáp án đúng lên top-1.
   - **Cảnh báo tác dụng phụ của Filter:** Dẫn chứng câu thăm dò *"Thi tự luận sinh viên được ra về sớm khi nào?"* (Điều 11 khoản 6) — khi bật filter `student`, đáp án lập tức biến mất hoàn toàn vì nằm trong tài liệu của cán bộ coi thi (`faculty`).
4. **Phút 6: Minh họa lỗi sai Điều ở Q4 (Đỗ Đức Đại)**
   - Minh họa Q4: Fixed, Recursive và Sentence đều lấy nhầm Điều 21 (chấm tự luận), chỉ có Heading v2 đưa được Điều 24 (chấm tiểu luận) vào top-3.
5. **Phút 7-8: Kết luận & Hỏi đáp**
   - Trả lời các câu hỏi phản biện của giảng viên theo tài liệu chuẩn bị sẵn.

---

## 6. Lệnh kiểm tra và nghiệm thu

Mọi kiểm tra đều có thể thực hiện trực tiếp trong terminal:

```bash
# 1. Kích hoạt môi trường ảo và chạy toàn bộ unit tests
.venv\Scripts\pytest tests/ -v

# 2. Kiểm tra bộ dữ liệu Khảo thí IUH đạt chuẩn Checkpoint 2
.venv\Scripts\python scripts/verify_corpus.py data/khao-thi-phuc-khao

# 3. Chạy benchmark đối chiếu toàn nhóm và cập nhật các file log
.venv\Scripts\python bench.py

# 4. Chạy demo thủ công KnowledgeBaseAgent
.venv\Scripts\python main.py
```

Toàn bộ repo hiện tại đạt trạng thái hoàn thiện tuyệt đối, không có conflict, đúng quy cách giảng dạy và thang điểm rubric của học phần.
