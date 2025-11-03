# Sử dụng base image Python 3.10-slim để có kích thước nhỏ gọn
FROM python:3.10-slim

# Thiết lập biến môi trường để pip không cache, tiết kiệm dung lượng
ENV PIP_NO_CACHE_DIR=true

# Cài đặt các gói hệ thống cần thiết
# - ffmpeg: Công cụ tối quan trọng để xử lý video và audio
# - git: Cần thiết để cài đặt một số thư viện python từ repo
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tạo và chuyển đến thư mục làm việc của ứng dụng
WORKDIR /app

# Cài đặt các thư viện Python cần thiết
# Chúng ta sẽ cài đặt các thư viện chính trước
RUN pip install --no-cache-dir \
    torch \
    torchaudio 
# Lưu ý: Dòng trên cài đặt torch phiên bản CPU để đảm bảo tính tương thích
# cao nhất (cho Windows, Linux, Mac). Chúng ta sẽ thảo luận về MPS/GPU sau.

# Cài đặt các thư viện AI và tiện ích
RUN pip install --no-cache-dir \
    openai-whisper \
    transformers \
    sentencepiece \
    piper-tts \
    pydub

# (Tùy chọn) Tải trước các mô hình để Docker image chứa sẵn
# RUN python -c "import whisper; whisper.load_model('base')"
# RUN python -c "from transformers import MarianMTModel, MarianTokenizer; \
#               MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-vi'); \
#               MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-vi')"
# Ghi chú: Việc tải trước mô hình sẽ làm image rất nặng.
# Tạm thời chúng ta sẽ để mã Python tự tải khi chạy lần đầu.

# Sao chép mã nguồn ứng dụng của chúng ta vào container
# Tạm thời chúng ta sẽ tạo một tệp main.py trống
COPY . .

# Lệnh mặc định khi chạy container (chúng ta sẽ chạy script chính)
CMD ["python", "main.py"]



