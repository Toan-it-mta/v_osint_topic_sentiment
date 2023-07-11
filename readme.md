# Các bước sử dụng Phân tích cảm xúc Version 2

- Hướng dẫn cài đặt các thư viện để chạy được mô hình Phân tích cảm xúc Version 2

# 1. Clone Project

- **Chạy câu lệnh:** git clone https://github.com/Toan-it-mta/v_osint_topic_sentiment.git

# 2. Hướng dẫn Download Mô Hình và cài đặt các thư viện cần thiết

## 2.1. Cài thư viện:

- **Chạy câu lệnh:** pip install -r ./requirements.txt

## 2.2. Download mô hình:

- **Chạy câu lệnh:** cd v_osint_toppic_sentiment
- **Chạy câu lệnh:** python download_model

# 3. Hướng dẫn sử dụng thư viện

from v_osint_topic_sentiment.sentiment_analysis import topic_sentiment_classification

text = """Người đứng đầu Bộ Quốc phòng tuyên bố rằng một thoả thuận hợp tác về mua sắm quốc phòng sẽ được ký với Bộ trưởng Quốc phòng Hoa Kỳ Lloyd Austin trong cuộc họp của họ vào thứ Sáu."""

predict_out = topic_sentiment_classification(text)

print(predict_out)

==============================================================================

{'sentiment_label': 'tich_cuc'}
