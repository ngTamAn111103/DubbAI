from pydub.silence import detect_nonsilent
import json
from pydub import AudioSegment
import os
import sys
import subprocess
import whisper
script_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(script_path)
SOURCE_FOLDER = os.path.join(BASE_DIR, "source")
VIDEO_INPUT_NAME = "test1.mp4"
AUDIO_OUTPUT_NAME = "original_audio.wav" # Bước 1 +2
VIDEO_INPUT_PATH = os.path.join(SOURCE_FOLDER, VIDEO_INPUT_NAME)
AUDIO_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, AUDIO_OUTPUT_NAME)
TRANSCRIPT_OUTPUT_NAME = "original_transcript.json" # Bước 3
TRANSCRIPT_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, TRANSCRIPT_OUTPUT_NAME)
WHISPER_MODEL_NAME = "large-v3" # Mô hình mạnh nhất
WHISPER_OPTIONS = {
    # Chúng ta sẽ tắt fp16 trong hàm, dựa trên device
    "word_timestamps": True,              # Rất quan trọng: Lấy mốc thời gian của TỪNG TỪ
    "condition_on_previous_text": True, # Giữ ngữ cảnh, giảm lặp (Mỗi tiếng anh ~~ True)
    "suppress_tokens": "-1",            # Đảm bảo timestamp được tạo ra
    "no_speech_threshold": 0.5,         # Ngưỡng phát hiện không có tiếng nói (linh hoạt hơn)
}
VAD_OPTIONS = {
    "min_silence_len": 1000, # (ms) Khoảng lặng tối thiểu để tính là "im lặng"
    "silence_thresh": -32,   # (dBFS) Ngưỡng âm lượng, càng thấp càng nhạy
    "keep_silence": 200      # (ms) Giữ lại một chút im lặng ở đầu/cuối
}
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
        
        print(f"✅ Bước 1 và 2 đã hoàn thành! Âm thanh đã được tách và lưu tại:")
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
def merge_short_segments(segments: list[dict], max_gap_sec: float = 1.5, min_segment_len_sec: float = 2.0) -> list[dict]:
    """
    Hợp nhất các segment ngắn dựa trên khoảng lặng và độ dài.
    """
    if not segments:
        return []

    print(f"\nBắt đầu Hợp nhất: có {len(segments)} segments ban đầu.")
    
    merged_segments = []
    
    # Bắt đầu với segment đầu tiên
    current_segment = segments[0].copy() 
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        
        # Tính khoảng lặng giữa 2 segment
        gap = next_segment['start'] - current_segment['end']
        
        # Tính thời lượng của segment hiện tại
        current_duration = current_segment['end'] - current_segment['start']
        
        # Kiểm tra điều kiện để gộp
        # 1. Khoảng lặng giữa chúng đủ nhỏ (ví dụ: < 1.5s)
        # 2. VÀ segment hiện tại quá ngắn (ví dụ: < 2s)
        if gap <= max_gap_sec and current_duration <= min_segment_len_sec:
            # Gộp!
            # Nối văn bản
            current_segment['text'] += " " + next_segment['text']
            # Cập nhật thời gian kết thúc
            current_segment['end'] = next_segment['end']
            print(f"   -> Đã gộp ID {current_segment['id']} và {next_segment['id']}")
        else:
            # Không gộp, lưu segment hiện tại
            merged_segments.append(current_segment)
            # Bắt đầu segment mới
            current_segment = next_segment.copy()
            
    # Đừng quên lưu segment cuối cùng!
    merged_segments.append(current_segment)
    
    print(f"✅ Hợp nhất hoàn tất: còn {len(merged_segments)} segments.")
    return merged_segments


def transcribe_audio(audio_path: str, model_name: str, device: str) -> dict | None:
    """
    Phiên âm âm thanh bằng Whisper, để mô hình tự xử lý VAD và segmentation.
    Hàm này sẽ trả về toàn bộ kết quả của Whisper, bao gồm cả "word-level timestamps".
    
    Args:
        audio_path: Đường dẫn đến tệp .wav
        model_name: Tên mô hình (ví dụ: "medium.en")
        device: Thiết bị chạy ("cpu", "cuda", "mps")

    Returns:
        Toàn bộ đối tượng `result` của Whisper, hoặc None nếu lỗi.
    """
    print(f"\nBắt đầu Bước 3: Phiên âm (Whisper-native VAD)...")
    try:
        # 1. Tải mô hình
        # print(f"Đang tải mô hình Whisper '{model_name}' trên '{device}'...")
        model = whisper.load_model(model_name, device=device)
        # print("Tải mô hình hoàn tất.")

        # 2. Định cấu hình tùy chọn
        transcribe_options = WHISPER_OPTIONS.copy()
        
        # Tự động quyết định dùng fp16 (tăng tốc) nếu không phải CPU
        # fp16 không được hỗ trợ tốt trên CPU
        transcribe_options["fp16"] = (device != "cpu")
        
        # Chọn ngôn ngữ (rất quan trọng nếu sau này nâng cấp)
        # Báo cho Whisper biết đây là tiếng Anh
        transcribe_options["language"] = "en" # Tối ưu cho englist
        transcribe_options["task"] = "transcribe" # Chỉ phiên âm, không dịch

        # print(f"Đang phiên âm '{audio_path}' với các tùy chọn:")
        # print(f"   {transcribe_options}")

        # 3. Chạy phiên âm (Đây là bước chính)
        # Đưa toàn bộ tệp âm thanh vào, không cần cắt
        result = model.transcribe(audio_path, **transcribe_options)
        
        print(f"✅ Bước 3 hoàn thành!")
        # print(f"   Phát hiện ngôn ngữ: {result.get('language')}")
        # print(f"   Tìm thấy {len(result.get('segments', []))} segments.")

        return result

    except Exception as e:
        print(f"❌ LỖI trong quá trình phiên âm: {e}")
        # In thêm chi tiết lỗi nếu có (ví dụ: lỗi CUDA)
        import traceback
        traceback.print_exc()
        return None
    
def main():
    device = "cpu" 
    
    # Bước 1 + 2: Tách âm thanh
    if not os.path.exists(AUDIO_OUTPUT_PATH):
        extracted_audio_file = extract_audio(VIDEO_INPUT_PATH, AUDIO_OUTPUT_PATH)
        if extracted_audio_file is None: sys.exit(1)
        
    # Bước 3: Phiên âm
    if not os.path.exists(TRANSCRIPT_OUTPUT_PATH):
        print("Bước 3: Phiên âm âm thanh gốc thành văn bản")
        segments = transcribe_audio(AUDIO_OUTPUT_PATH, WHISPER_MODEL_NAME, device)

        # Phiên âm không thành công: Dừng chương trình
        if segments is None: return

        # Hợp nhất các đoạn ngắn
        segments = merge_short_segments(segments, max_gap_sec=1.5, min_segment_len_sec=2.0)

        # Lưu tệp json
        try:
            with open(TRANSCRIPT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"❌ LỖI: Không thể lưu tệp JSON phiên âm: {e}")
            sys.exit(1)
        

    print("Sẵn sàng cho Bước 4 (Dịch thuật)")
        
main()