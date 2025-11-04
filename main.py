import subprocess
import sys
import os
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
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

# Tệp JSON chứa kết quả phiên âm
TRANSCRIPT_OUTPUT_NAME = "original_transcript.json"
TRANSCRIPT_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, TRANSCRIPT_OUTPUT_NAME)
# Tệp JSON dịch thuật Anh -> Việt
TRANSLATED_TRANSCRIPT_NAME = "translated_transcript.json"
TRANSLATED_TRANSCRIPT_PATH = os.path.join(SOURCE_FOLDER, TRANSLATED_TRANSCRIPT_NAME)

# Cấu hình mô hình
WHISPER_MODEL_NAME = "base"
# Mô hình dịch thuật
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"
# ----------------------------------------

def get_device() -> str:
    """Kiểm tra và trả về thiết bị (device) phù hợp cho PyTorch."""
    if torch.cuda.is_available():
        print("Phát hiện GPU CUDA. Đang sử dụng 'cuda'.")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Phát hiện Apple Silicon (M-series). Đang sử dụng 'mps'.")
        return "mps"
    else:
        print("Không phát hiện GPU/MPS. Đang sử dụng 'cpu'.")
        return "cpu"
    
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

def transcribe_audio(audio_path: str, model_name: str, device: str) -> list[dict] | None:
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

    try:
        model = whisper.load_model(model_name, device=device)
        use_fp16 = (device != "cpu")
        result = model.transcribe(audio_path, fp16=use_fp16, task="transcribe")
        detected_lang = result.get('language', 'không rõ')
        return result['segments']
    except Exception as e:
        print(f"❌ LỖI trong quá trình phiên âm: {e}")
        return None
    
def translate_segments(segments: list[dict], model_name: str, device: str) -> list[dict] | None:
    """
    Dịch văn bản trong các segments sang tiếng Việt.
    
    Args:
        segments: Danh sách segments từ Whisper (chứa 'start', 'end', 'text').
        model_name: Tên mô hình dịch trên Hugging Face.
        device: Thiết bị để chạy (cpu, cuda, mps).

    Returns:
        Danh sách segments mới với 'text' đã được dịch.
    """
    
    try:
        
        # PyTorch index cho thiết bị (0 cho cuda/mps, -1 cho cpu)
        torch_device_index = 0 if device in ["cuda", "mps"] else -1
        translator = pipeline("translation", 
                              model=model_name, 
                              device=torch_device_index)

        # 2. Chuẩn bị dữ liệu (dịch theo batch cho nhanh)
        # Lấy văn bản (đã loại bỏ khoảng trắng thừa) từ mỗi segment
        texts_to_translate = [segment['text'].strip() for segment in segments]
        
        # 3. Thực hiện dịch
        translated_results = translator(texts_to_translate, batch_size=16) # batch_size=16

        # 4. Tạo lại danh sách segments với văn bản đã dịch
        translated_segments = []
        for i, segment in enumerate(segments):
            translated_text = translated_results[i]['translation_text']
            
            new_segment = {
                "id": segment['id'],
                "start": segment['start'],
                "end": segment['end'],
                "original_text": segment['text'], # Giữ lại văn bản gốc để tham khảo
                "text": translated_text  # Thay thế bằng văn bản đã dịch
            }
            translated_segments.append(new_segment)
            
        # print(f"✅ Bước 4 hoàn thành!")
        return translated_segments
        
    except Exception as e:
        print(f"❌ LỖI trong quá trình dịch thuật: {e}")
        return None
    
# ---- Hàm chính để chạy ứng dụng ----
def main():
    # Xác định thiết bị chạy AI (chạy 1 lần ở đầu)
    device = get_device()

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
    if os.path.exists(TRANSCRIPT_OUTPUT_PATH):
        print(f"\nĐã tìm thấy tệp phiên âm: {TRANSCRIPT_OUTPUT_PATH}. Bỏ qua Bước 3.")
        with open(TRANSCRIPT_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    else:
        segments = transcribe_audio(AUDIO_OUTPUT_PATH, WHISPER_MODEL_NAME, device)
        if segments is None: sys.exit(1)
        
        # Lưu tệp JSON
        print(f"\nĐang lưu kết quả phiên âm vào '{TRANSCRIPT_OUTPUT_PATH}'...")
        try:
            with open(TRANSCRIPT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=4, ensure_ascii=False)
            print("✅ Đã lưu phiên âm thành công.")
        except Exception as e:
            print(f"❌ LỖI: Không thể lưu tệp JSON phiên âm: {e}")
            sys.exit(1)


    # In ra 3 segment đầu tiên để kiểm tra
    print("\n--- Kết quả phiên âm (3 segment đầu tiên) ---")
    for i, segment in enumerate(segments[:3]):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")
    print("---------------------------------------------")

    # Bước 4: Dịch thuật
    translated_segments = translate_segments(segments, TRANSLATION_MODEL_NAME, device)
    if translated_segments is None: sys.exit(1)
        
    # Lưu tệp JSON đã dịch
    # print(f"\nĐang lưu kết quả dịch thuật vào '{TRANSLATED_TRANSCRIPT_PATH}'...")
    try:
        with open(TRANSLATED_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
            # ensure_ascii=False RẤT QUAN TRỌNG để lưu tiếng Việt
            json.dump(translated_segments, f, indent=4, ensure_ascii=False)
        # print("✅ Đã lưu dịch thuật thành công.")
    except Exception as e:
        print(f"❌ LỖI: Không thể lưu tệp JSON dịch thuật: {e}")
        sys.exit(1)

    # In ra 3 segment đã dịch đầu tiên để kiểm tra
    print("\n--- Kết quả dịch thuật (3 segment đầu tiên) ---")
    for i, segment in enumerate(translated_segments[:3]):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")
    print("-------------------------------------------------")
    
    print("\n(Tạm thời kết thúc - Các bước tiếp theo sẽ được xây dựng)")
    print("-----------------------------------")



if __name__ == "__main__":
    main()