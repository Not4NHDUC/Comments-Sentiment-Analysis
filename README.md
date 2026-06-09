# Phân Loại Cảm Xúc Đánh Giá Sản Phẩm Tiếng Việt

Môn: Học Máy & Khai Phá Dữ Liệu — HUST  


## Mô tả
Hệ thống ML/NLP tự động phân loại cảm xúc đánh giá sản phẩm tiếng Việt  
từ các sàn TMĐT (Tiki, Shopee, Lazada) thành 3 nhãn: **Positive / Neutral / Negative**.

## Kết quả
| Model | Feature | F1-macro |
|-------|---------|----------|
| **SVM** | **TF-IDF** | **0.7279** |
| XGBoost | BoW | 0.7005 |
| XGBoost | TF-IDF | 0.6772 |
| NaiveBayes | BoW | 0.6645 |

## Cấu trúc dự án
```
├── crawlers/          # Thu thập dữ liệu (Tiki, Shopee, Lazada)
├── data/
│   ├── raw/           # Dữ liệu thô từ crawler
│   └── processed/     # Dữ liệu đã xử lý
├── src/
│   ├── preprocessing/ # Pipeline tiền xử lý tiếng Việt
│   └── models/        # Train & evaluate models
├── models/            # Model đã train (.pkl)
├── reports/           # Kết quả, biểu đồ, confusion matrix
├── app/
│   ├── main.py        # FastAPI
│   └── dashboard.py   # Streamlit Dashboard
└── main.py            # Entry point crawler
```

## Cài đặt
```bash
pip install -r requirements.txt
playwright install chromium
```

## Chạy từng phase

### Phase 1 — Thu thập dữ liệu
```bash
python main.py --platforms tiki --target 800
```

### Phase 2 — Tiền xử lý
```bash
python src/run_preprocessing.py
```

### Phase 3 — Huấn luyện mô hình
```bash
python src/run_training.py
```

### Phase 4 — Demo
```bash
# Terminal 1 - API
uvicorn app.main:app --reload

# Terminal 2 - Dashboard
streamlit run app/dashboard.py
```

> **Lưu ý:** Model PhoBERT (~500MB) không được include trong repo do giới hạn GitHub.  
> Để dùng PhoBERT, chạy trước: `python src/models/train_phobert.py` (~15-20 phút, cần GPU CUDA)  
> Nếu không có GPU, dashboard vẫn hoạt động bình thường với model TF-IDF+SVM.

Mở trình duyệt: http://localhost:8501

## Tech Stack
- **Crawler**: requests, Playwright
- **NLP**: underthesea, TF-IDF, BoW
- **ML**: scikit-learn (SVM, RF, NB), XGBoost
- **API**: FastAPI + Uvicorn
- **Dashboard**: Streamlit
