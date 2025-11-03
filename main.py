import subprocess
import sys
import os
import torch
import whisper
import json

# ---- Cấu hình đường dẫn ----
# Lấy đường dẫn tuyệt đối của tệp script này (BÊN TRONG container)
# ví dụ: /app/main.py
script_path = os.path.abspath(__file__)

# Lấy đường dẫn thư mục cha chứa tệp script
# BASE_DIR sẽ là: /app
BASE_DIR = os.path.dirname(script_path)

# Xây dựng các đường dẫn khác dựa trên BASE_DIR
SOURCE_FOLDER = os.path.join(BASE_DIR, "source")
VIDEO_INPUT_NAME = "test1.mp4"
AUDIO_OUTPUT_NAME = "original_audio.wav"

VIDEO_INPUT_PATH = os.path.join(SOURCE_FOLDER, VIDEO_INPUT_NAME)
AUDIO_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, AUDIO_OUTPUT_NAME)

# MỚI: Tệp JSON chứa kết quả phiên âm
TRANSCRIPT_OUTPUT_NAME = "original_transcript.json"
TRANSCRIPT_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, TRANSCRIPT_OUTPUT_NAME)

# Cấu hình mô hình
WHISPER_MODEL_NAME = "base"
# ----------------------------------------


def extract_audio(video_input_path: str, audio_output_path: str) -> str | None:
    """
    Sử dụng ffmpeg để tách âm thanh từ tệp video đầu vào.
    
    Chúng ta sẽ chuyển đổi âm thanh thành định dạng WAV, 16kHz, mono.
    Đây là định dạng tối ưu cho các mô hình AI Speech-to-Text như Whisper.
    
    Args:
        video_input_path: Đường dẫn đến tệp video .mp4 (ví dụ: /app/source/input_video.mp4)
        audio_output_path: Đường dẫn lưu tệp âm thanh .wav (ví dụ: /app/source/original_audio.wav)

    Returns:
        Trả về đường dẫn tệp âm thanh nếu thành công, ngược lại trả về None.
    """
    # print(f"Bắt đầu Bước 1: Tách âm thanh từ '{video_input_path}'...")
    
    # Xây dựng lệnh ffmpeg
    # -i : Tệp đầu vào
    # -vn : Bỏ qua video (no video)
    # -acodec pcm_s16le : Định dạng âm thanh là WAV 16-bit
    # -ar 16000 : Tần số lấy mẫu 16kHz (tốt nhất cho Whisper)
    # -ac 1 : 1 kênh âm thanh (mono)
    # -y : Tự động ghi đè tệp đầu ra nếu đã tồn tại
    command = [
        'ffmpeg',
        '-i', video_input_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',
        audio_output_path
    ]
    
    try:
        # Chạy lệnh
        # capture_output=True: Lấy stdout và stderr
        # text=True: Giải mã stdout/stderr thành text (thay vì bytes)
        # check=True: Tự động ném lỗi (raise Exception) nếu ffmpeg trả về mã lỗi
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        print(f"✅ Bước 1 hoàn thành! Âm thanh đã được tách và lưu tại:")
        print(f"   {audio_output_path}")
        return audio_output_path
        
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy 'ffmpeg'. Hãy đảm bảo nó đã được cài đặt trong Dockerfile.")
        return None
    except subprocess.CalledProcessError as e:
        # Nếu ffmpeg chạy bị lỗi (ví dụ: không tìm thấy file input)
        print(f"❌ LỖI: ffmpeg thất bại với mã lỗi {e.returncode}")
        print("   Lỗi chi tiết (stderr):")
        print(f"   {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ LỖI không xác định: {e}")
        return None

def transcribe_audio(audio_path: str, model_name: str) -> list[dict] | None:
    """
    Sử dụng Whisper để phiên âm âm thanh và lấy mốc thời gian.
    
    Args:
        audio_path: Đường dẫn đến tệp âm thanh .wav
        model_name: Tên mô hình Whisper (ví dụ: "base", "small", "medium")

    Returns:
        Một danh sách (list) các 'segments'. 
        Ví dụ: [
            {'start': 0.0, 'end': 5.2, 'text': ' Hello world.'},
            {'start': 5.2, 'end': 8.0, 'text': ' This is a test.'}
        ]
        Trả về None nếu có lỗi.
    """

    # 1. Tự động phát hiện thiết bị (CPU, GPU-CUDA, Mac-MPS)
    # Điều này thực hiện yêu cầu của bạn là hỗ trợ đa nền tảng
    if torch.cuda.is_available():
        device = "cuda"
        print("Phát hiện GPU CUDA. Đang sử dụng 'cuda'.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Phát hiện Apple Silicon (M-series). Đang sử dụng 'mps'.")
    else:
        device = "cpu"
        print("Không phát hiện GPU/MPS. Đang sử dụng 'cpu'.")
        
    try:
        # 2. Tải mô hình
        # Lần chạy đầu tiên sẽ tải mô hình về (sẽ mất một lúc)
        print(f"Đang tải mô hình Whisper '{model_name}' (lần đầu có thể mất vài phút)...")
        model = whisper.load_model(model_name, device=device)

        # 3. Thực hiện phiên âm
        # 'fp16=False' sẽ an toàn hơn cho CPU, nhưng chậm hơn
        use_fp16 = (device != "cpu")
        print(f"Đang phiên âm (sử dụng fp16={use_fp16})...")
        
        # task="transcribe" sẽ chỉ trả về ngôn ngữ gốc
        result = model.transcribe(audio_path, fp16=use_fp16, task="transcribe")
        
        # 4. Trích xuất thông tin
        detected_lang = result.get('language', 'không rõ')
        # print(f"✅ Bước 3 hoàn thành! Ngôn ngữ phát hiện: {detected_lang.upper()}")
        
        # Đây chính là phần bạn cần: danh sách các segment với mốc thời gian
        segments = result['segments']
        return segments

    except Exception as e:
        print(f"❌ LỖI trong quá trình phiên âm: {e}")
        return None
    
# ---- Hàm chính để chạy ứng dụng ----
def main():
    
    # Kiểm tra xem tệp video đầu vào có tồn tại không
    if not os.path.exists(VIDEO_INPUT_PATH):
        print(f"❌ LỖI: Không tìm thấy tệp video đầu vào tại:")
        print(f"   {VIDEO_INPUT_PATH}")
        print("   Hãy đảm bảo bạn đã đặt video vào thư mục 'source' và đặt tên là 'input_video.mp4'")
        sys.exit(1) # Thoát chương trình với mã lỗi
        
    # Bước 1 + 2: Tách âm thanh
    extracted_audio_file = extract_audio(VIDEO_INPUT_PATH, AUDIO_OUTPUT_PATH)
    
    if extracted_audio_file is None:
        print("Dừng chương trình do lỗi ở Bước 1.")
        sys.exit(1)
        
    # Bước 3: Phiên âm (Audio to Text)
    segments = transcribe_audio(AUDIO_OUTPUT_PATH, WHISPER_MODEL_NAME)
    if segments is None:
        print("Dừng chương trình do lỗi ở Bước 3.")
        sys.exit(1)
    else:
        try:
            with open(TRANSCRIPT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                # indent=4: Giúp tệp JSON dễ đọc (pretty print)
                # ensure_ascii=False: Hỗ trợ ký tự Unicode (quan trọng nếu 
                #                     ngôn ngữ gốc có dấu, ví dụ tiếng Pháp, TBN...)
                json.dump(segments, f, indent=4, ensure_ascii=False)
            print("✅ Đã lưu phiên âm thành công.")
        except Exception as e:
            print(f"❌ LỖI: Không thể lưu tệp JSON phiên âm: {e}")
            sys.exit(1) # Nếu không lưu được, chúng ta cũng nên dừng lại


    # In ra 3 segment đầu tiên để kiểm tra
    print("\n--- Kết quả phiên âm (3 segment đầu tiên) ---")
    for i, segment in enumerate(segments[:3]):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")
    print("---------------------------------------------")

if __name__ == "__main__":
    main()