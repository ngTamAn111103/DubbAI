# Thư viện
from dotenv import load_dotenv
import time
import torch
import os
import subprocess
import sys
import whisper
import json
from transformers import MarianMTModel, MarianTokenizer,pipeline 
import math
import pprint

# Bắt đầu tính giờ
start_time = time.time()
load_dotenv()

# Cấu hình hạn chế đụng vào
script_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(script_path)
AUDIO_OUTPUT_NAME = os.getenv("AUDIO_OUTPUT_NAME")
VIDEO_INPUT_NAME = os.getenv("VIDEO_INPUT_NAME")
SOURCE_FOLDER = os.path.join(BASE_DIR, "source")
AUDIO_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, AUDIO_OUTPUT_NAME)
VIDEO_INPUT_PATH = os.path.join(SOURCE_FOLDER, VIDEO_INPUT_NAME)
TRANSCRIPT_OUTPUT_NAME = "original_transcript.json"
TRANSCRIPT_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, TRANSCRIPT_OUTPUT_NAME)
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME")
TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL_NAME")
TTS_DATA_NAME = os.getenv("TTS_DATA_NAME")
TRANSLATED_TRANSCRIPT_NAME = "translated_transcript.json" # Bước 4
TRANSLATED_TRANSCRIPT_PATH = os.path.join(SOURCE_FOLDER, TRANSLATED_TRANSCRIPT_NAME)
TTS_DATA_PATH = os.path.join(SOURCE_FOLDER, TTS_DATA_NAME)

def to_bool(value: str) -> bool:
    """
    Hàm trợ giúp để chuyển string "True" -> boolean True
    """
    return str(value).lower() in ['true', '1', 't', 'yes']

# Đọc và TỰ CHUYỂN ĐỔI kiểu dữ liệu
WHISPER_OPTIONS = {
    # .getenv("KEY", "default_value")
    "word_timestamps": to_bool(os.getenv("WHISPER_WORD_TIMESTAMPS", "True")),
    "condition_on_previous_text": to_bool(os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "True")),
    "suppress_tokens": os.getenv("WHISPER_SUPPRESS_TOKENS", "-1"),
    "no_speech_threshold": float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", 0.5)),
}

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
        model = whisper.load_model(model_name, device=device)

        # 2. Định cấu hình tùy chọn
        transcribe_options = WHISPER_OPTIONS.copy()
        
        # Tự động quyết định dùng fp16 (tăng tốc) nếu không phải CPU
        # fp16 không được hỗ trợ tốt trên CPU
        transcribe_options["fp16"] = (device != "cpu")
        
        # Chọn ngôn ngữ (rất quan trọng nếu sau này nâng cấp)
        # Báo cho Whisper biết đây là tiếng Anh
        if WHISPER_OPTIONS["condition_on_previous_text"]:
            transcribe_options["language"] = "en" # Tối ưu cho englist

        transcribe_options["task"] = "transcribe" # Chỉ phiên âm, không dịch

        # 3. Chạy phiên âm (Đây là bước chính)
        # Đưa toàn bộ tệp âm thanh vào, không cần cắt
        result = model.transcribe(audio_path, **transcribe_options)
        print(f"✅ Bước 3 hoàn thành!")
        return result

    except Exception as e:
        print(f"❌ LỖI trong quá trình phiên âm: {e}")
        # In thêm chi tiết lỗi nếu có (ví dụ: lỗi CUDA)
        import traceback
        traceback.print_exc()
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

def translate_segments(whisper_result: dict, model_name: str, device: str, batch_size: int = 8 ) -> dict | None:
    """
    Dịch các segment văn bản từ Anh sang Việt, giữ nguyên cấu trúc dict.
    Tối ưu hóa để sử dụng bộ nhớ ổn định (KHẮC PHỤC RÒ RỈ BỘ NHỚ MPS/CUDA).

    Args:
        whisper_result (dict): json phiên âm (text englist)
        model_name (str): Model dịch thuật (Helsinki-NLP/opus-mt-en-vi, ...)
        device (str): cuda, mps, cpu

    Returns:
        dict | None: json 
    """
    try:
        # 1. Tải mô hình và tokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.to(device)

        # 2. Tạo bản sao dữ liệu NGAY TỪ ĐẦU
        segments_to_translate = whisper_result.get('segments', [])
        if not segments_to_translate:
            print("❌ LỖI: Không tìm thấy 'segments' trong dữ liệu đầu vào.")
            return None
            
        translated_data = json.loads(json.dumps(whisper_result))
        
        # Chỉ lấy đoạn text tiếng anh
        texts_to_translate = [seg['text'].strip() for seg in segments_to_translate]
        total_segments = len(texts_to_translate)

        # 3. Logic batching
        total_batches = math.ceil(total_segments / batch_size)
        
        print(f"Chuẩn bị dịch {total_segments} segments thành {total_batches} batches ({batch_size} segments/batch)...")

        for i in range(0, total_segments, batch_size):
            batch_texts = texts_to_translate[i : i + batch_size]
            
            # --- BẮT ĐẦU KHỐI TỐN BỘ NHỚ ---
            batch_tokens = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            translated_tokens = model.generate(**batch_tokens)
            
            translated_batch_texts = [
                tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens
            ]
            # --- KẾT THÚC KHỐI TỐN BỘ NHỚ ---

            # Cập nhật trực tiếp vào 'translated_data'
            for j in range(len(translated_batch_texts)):
                segment_index = i + j
                translated_data['segments'][segment_index]['original_text'] = translated_data['segments'][segment_index]['text']
                translated_data['segments'][segment_index]['text'] = translated_batch_texts[j]

            # IN LOG TIẾN ĐỘ
            current_batch_num = (i // batch_size) + 1
            segments_done = min(i + batch_size, total_segments)
            print(f"   ... Đã dịch xong batch {current_batch_num} / {total_batches} (Hoàn thành {segments_done}/{total_segments} segments)")

            # ==========================================================
            # === DỌN DẸP BỘ NHỚ THỦ CÔNG (RẤT QUAN TRỌNG) ===
            del batch_tokens
            del translated_tokens
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            # ==========================================================

        print("Dịch thuật hoàn tất.")

        # 4. Trả về kết quả
        print(f"✅ Bước 4 hoàn thành!")
        return translated_data
        
    except Exception as e:
        print(f"❌ LỖI trong quá trình dịch thuật: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_tts_data_file(translated_segments: list[dict], output_script_path: str):
    """
    Tạo mảng dữ liệu TTS và GHI NỘI DUNG MẢNG đó ra tệp.
    Đồng thời trả về danh sách segments đã cập nhật cho Bước 6.
    """
    # print(f"\nBắt đầu Bước 5: Ghi mảng dữ liệu TTS vào '{output_script_path}'...")
    
    tts_data_list = []
    segments_with_audio_path = []
    
    try:
        # Lặp qua các segment đã dịch
        for segment in translated_segments['segments']:
            segment_id = segment['id']
            text_to_speak = segment['text'].strip()
            
            # 1. Tạo dữ liệu cho mảng
            formatted_text = f"[KienThucQuanSu]{text_to_speak}"
            audio_output_file = f"audio_VN/{segment_id}.wav"
            
            tts_tuple = (formatted_text, audio_output_file)
            tts_data_list.append(tts_tuple)
            
            # 2. Cập nhật segment cho Bước 6
            segment['audio_path'] = os.path.join(SOURCE_FOLDER, audio_output_file)
            segments_with_audio_path.append(segment)

        # 3. Ghi mảng (dưới dạng chuỗi) ra tệp
        with open(output_script_path, 'w', encoding='utf-8') as f:
            # Sử dụng pprint.pformat để tạo chuỗi Python đẹp
            # indent=4 và width=120 (để tránh ngắt dòng quá sớm)
            # Sẽ tạo ra định dạng giống hệt ví dụ của bạn
            file_content = pprint.pformat(tts_data_list, indent=4, width=120)
            
            # Ghi vào tệp. 
            # (Bạn có thể thêm `tts_data = ` ở đầu nếu muốn nó là tệp .py)
            # f.write("tts_data = ")
            f.write(file_content)
            # f.write("\n") 

        print(f"✅ Bước 5 hoàn thành! Đã ghi mảng dữ liệu vào tệp.")
        # Trả về danh sách segment đã cập nhật cho Bước 6
        return segments_with_audio_path

    except Exception as e:
        print(f"❌ LỖI trong quá trình ghi tệp dữ liệu TTS: {e}")
        return None
def main():

    # --- Bước 1 & 2: Tách âm thanh ---
    # Nếu chưa có file .wav
    if not os.path.exists(AUDIO_OUTPUT_PATH):
        extracted_audio_file = extract_audio(VIDEO_INPUT_PATH, AUDIO_OUTPUT_PATH)
        if extracted_audio_file is None: 
            sys.exit(1)
    

    # --- Bước 3: Phiên âm ---
    # Nếu chưa có file phiên âm [original_transcript].json
    if not os.path.exists(TRANSCRIPT_OUTPUT_PATH):
        whisper_result = transcribe_audio(AUDIO_OUTPUT_PATH, WHISPER_MODEL_NAME, get_device())

        # Phiên âm không thành công: Dừng chương trình
        if whisper_result is None: 
            print("Lỗi: Phiên âm thất bại.")
            sys.exit(1)

        # 2. Trích xuất danh sách (LIST) segments từ DICT
        segments_list = whisper_result.get('segments')
        if not segments_list:
            print("Lỗi: Whisper không tìm thấy đoạn âm thanh nào.")
            sys.exit(1)

        # 3. Truyền danh sách (LIST) vào hàm merge
        #    Hàm này sẽ trả về một danh sách (LIST) đã được gộp
        merged_segments_list = merge_short_segments(segments_list, max_gap_sec=1.5, min_segment_len_sec=2.0)
        whisper_result['segments'] = merged_segments_list
        # 5. Lưu toàn bộ DICT (đã chứa segments được gộp)
        try:
            with open(TRANSCRIPT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                # Lưu lại toàn bộ đối tượng whisper_result
                json.dump(whisper_result, f, indent=4, ensure_ascii=False)
            print(f"✅ Đã lưu phiên âm (ĐÃ GỘP) vào: {TRANSCRIPT_OUTPUT_PATH}")
        except Exception as e:
            print(f"❌ LỖI: Không thể lưu tệp JSON phiên âm: {e}")
            sys.exit(1)
    else:
        with open(TRANSCRIPT_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            segments = json.load(f)





main()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Thời gian Docker: {elapsed_time:.2f} giây")
