import subprocess
import sys
import os
import torch
import whisper
from transformers import pipeline 
from pydub import AudioSegment
import json
import pprint
import argparse


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
# Tệp chứa mảng dữ liệu TTS
# Chúng ta có thể lưu nó dưới dạng tệp .py để dễ import sau này
TTS_DATA_NAME = "tts_data.py" 
TTS_DATA_PATH = os.path.join(SOURCE_FOLDER, TTS_DATA_NAME)
# Tệp đầu ra cho Bước 6
FINAL_AUDIO_NAME = "dubbed_audio.wav"
FINAL_AUDIO_PATH = os.path.join(SOURCE_FOLDER, FINAL_AUDIO_NAME)
FINAL_VIDEO_NAME = "final_dubbed_video.mp4"
FINAL_VIDEO_PATH = os.path.join(SOURCE_FOLDER, FINAL_VIDEO_NAME)

# Cấu hình mô hình
WHISPER_MODEL_NAME = "medium.en"
# Mô hình dịch thuật
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"

# Cấu hình tùy chọn cho Whisper
# Đây là nơi bạn "tinh chỉnh" (tune) để sửa lỗi mốc thời gian
WHISPER_OPTIONS = {
    "no_speech_threshold": 0.3,  # Hạ thấp ngưỡng để dễ phát hiện im lặng hơn (Mặc định 0.6)
    "hallucination_silence_threshold": 3.0, # Xóa ảo giác trong khoảng lặng > 3 giây
    "word_timestamps": True,     # Bật để tăng độ chính xác của mốc thời gian
    "fp16": False                # Đặt là False nếu chạy trên CPU (an toàn)
}
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
        
        # Cập nhật tùy chọn fp16 dựa trên thiết bị
        transcribe_options = WHISPER_OPTIONS.copy()
        transcribe_options["fp16"] = (device != "cpu")

        # Sử dụng **để giải nén (unpack) dictionary vào các tham số
        result = model.transcribe(audio_path, task="transcribe", **transcribe_options)
        
        print(f"✅ Bước 3 hoàn thành! Ngôn ngữ: {result.get('language', 'không rõ')}")
        
        # In ra segment đầu tiên để kiểm tra mốc thời gian
        if result['segments']:
            seg0 = result['segments'][0]
            print(f"   Kiểm tra: Segment 0 bắt đầu từ {seg0['start']:.2f}s")
        
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
        for segment in translated_segments:
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
            f.write("tts_data = ")
            f.write(file_content)
            f.write("\n") 

        print(f"✅ Bước 5 hoàn thành! Đã ghi mảng dữ liệu vào tệp.")
        # Trả về danh sách segment đã cập nhật cho Bước 6
        return segments_with_audio_path

    except Exception as e:
        print(f"❌ LỖI trong quá trình ghi tệp dữ liệu TTS: {e}")
        return None
    
def apply_ffmpeg_atempo(input_segment: AudioSegment, speed: float, 
                        temp_dir: str = "/tmp") -> AudioSegment:
    """
    Sử dụng ffmpeg với bộ lọc 'atempo' để co/dãn âm thanh một cách an toàn.
    Hàm này xử lý các giới hạn 0.5-100.0 của atempo.
    """
    if abs(speed - 1.0) < 0.01:
        return input_segment # Không cần thay đổi

    # Tạo đường dẫn tệp tạm
    # Chúng ta phải lưu segment ra tệp để ffmpeg đọc
    temp_input = os.path.join(temp_dir, "temp_atempo_in.wav")
    temp_output = os.path.join(temp_dir, "temp_atempo_out.wav")
    
    input_segment.export(temp_input, format="wav")

    # Xây dựng chuỗi bộ lọc atempo
    # Ví dụ: speed = 0.3 -> [0.6, 0.5] (vì 0.5 * 0.6 = 0.3)
    # Ví dụ: speed = 0.2 -> [0.8, 0.5, 0.5] (vì 0.5 * 0.5 * 0.8 = 0.2)
    filters = []
    current_speed = speed
    
    # Xử lý tốc độ quá thấp (< 0.5)
    while current_speed < 0.5:
        filters.append("atempo=0.5")
        current_speed /= 0.5 # Tốc độ còn lại để áp dụng
    
    # Xử lý tốc độ quá cao (> 100.0)
    while current_speed > 100.0:
        filters.append("atempo=100.0")
        current_speed /= 100.0

    # Áp dụng phần tốc độ còn lại (ví dụ: 0.6, hoặc 1.5, hoặc 0.8)
    if abs(current_speed - 1.0) > 0.01:
        filters.append(f"atempo={current_speed}")

    # Nối các bộ lọc lại, ví dụ: "atempo=0.8,atempo=0.5,atempo=0.5"
    filter_chain = ",".join(filters)

    # Xây dựng và chạy lệnh ffmpeg
    command = [
        'ffmpeg',
        '-i', temp_input,
        '-filter:a', filter_chain,
        '-y', temp_output
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        # Tải tệp kết quả đã được co/dãn
        output_segment = AudioSegment.from_wav(temp_output)
        
        # Dọn dẹp tệp tạm
        os.remove(temp_input)
        os.remove(temp_output)
        
        return output_segment
        
    except Exception as e:
        print(f"   ❌ LỖI khi đang chạy atempo (tốc độ {speed:.2f}x): {e}")
        print(f"   ...Sử dụng segment gốc (không đồng bộ) thay thế.")
        # Dọn dẹp tệp tạm
        if os.path.exists(temp_input): os.remove(temp_input)
        if os.path.exists(temp_output): os.remove(temp_output)
        return input_segment # Trả về bản gốc nếu thất bại
    
    
def synchronize_and_combine(segments_with_audio_path: list[dict], 
                            final_audio_path: str) -> str | None:
    """
    (Bước 6.1) Đồng bộ (co/dãn) các tệp TTS và nối chúng lại.
    Phiên bản này sử dụng ffmpeg atempo thay vì pydub.speedup.
    """
    print(f"\nBắt đầu Bước 6.1: Đồng bộ và Nối các tệp âm thanh...")
    
    final_audio = AudioSegment.empty()
    last_segment_end_ms = 0.0 # Theo dõi mốc thời gian cuối cùng (tính bằng ms)
    
    try:
        # Lặp qua các segment đã có đường dẫn 'audio_path'
        for i, segment in enumerate(segments_with_audio_path):
            
            print(f"--- Đang xử lý segment {i} (ID: {segment['id']}) ---")
            
            target_start_ms = segment['start'] * 1000
            target_end_ms = segment['end'] * 1000
            target_duration_ms = target_end_ms - target_start_ms

            # 1. Xử lý khoảng lặng (Silence)
            if target_start_ms > last_segment_end_ms:
                silence_duration = target_start_ms - last_segment_end_ms
                final_audio += AudioSegment.silent(duration=silence_duration)
                print(f"   ... Thêm {silence_duration:.0f}ms khoảng lặng.")
                
            # 2. Tải tệp âm thanh TTS
            audio_file_path = segment['audio_path']
            if not os.path.exists(audio_file_path):
                print(f"   ⚠️ CẢNH BÁO: Không tìm thấy tệp {audio_file_path}. Bỏ qua segment.")
                last_segment_end_ms = target_end_ms
                continue

            tts_segment = AudioSegment.from_wav(audio_file_path)
            current_duration_ms = len(tts_segment)
            
            # 3. Đồng bộ thời gian (Time-Stretching)
            if target_duration_ms <= 0 or current_duration_ms <= 0:
                print(f"   ⚠️ CẢNH BÁO: Segment {i} có thời lượng không hợp lệ. Bỏ qua.")
                last_segment_end_ms = target_end_ms
                continue
            
            playback_speed = current_duration_ms / target_duration_ms

            print(f"   Đồng bộ segment {i}: {current_duration_ms:.0f}ms -> {target_duration_ms:.0f}ms (tốc độ {playback_speed:.2f}x)")

            # === KHỐI LOGIC MỚI (V3.0) ===
            # Gọi hàm helper ffmpeg atempo của chúng ta
            processed_segment = apply_ffmpeg_atempo(tts_segment, playback_speed)
            # === KẾT THÚC KHỐI LOGIC MỚI ===

            # 4. Nối âm thanh đã xử lý
            final_audio += processed_segment
            last_segment_end_ms = target_end_ms

        # 5. Lưu tệp âm thanh cuối cùng
        print(f"Đang lưu tệp âm thanh lồng tiếng cuối cùng tại: {final_audio_path}")
        final_audio.export(final_audio_path, format="wav")
        print(f"✅ Bước 6.1 hoàn thành!")
        return final_audio_path

    except Exception as e:
        print(f"❌ LỖI trong quá trình đồng bộ âm thanh: {e}")
        return None
    
    
def merge_audio_to_video(video_input_path: str, audio_input_path: str, 
                         video_output_path: str) -> str | None:
    """
    Ghép tệp âm thanh lồng tiếng vào video gốc (đã xóa tiếng).
    """
    
    # Lệnh ffmpeg
    # -i [video_input]: Video gốc
    # -i [audio_input]: Âm thanh lồng tiếng mới
    # -c:v copy: Sao chép luồng video, không encode lại (RẤT NHANH)
    # -map 0:v:0: Chọn luồng video từ file đầu vào (0)
    # -map 1:a:0: Chọn luồng audio từ file thứ hai (1) -> BỎ ÂM THANH GỐC
    # -shortest: Kết thúc video khi luồng ngắn nhất (video hoặc audio) kết thúc
    # -y: Ghi đè file đầu ra
    command = [
        'ffmpeg',
        '-i', video_input_path,
        '-i', audio_input_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y',
        video_output_path
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Bước 6.2 hoàn thành! Video lồng tiếng đã được lưu tại:")
        print(f"   {video_output_path}")
        return video_output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ LỖI: ffmpeg thất bại khi ghép video: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ LỖI không xác định khi ghép video: {e}")
        return None
    

# def main():
#     # Xác định thiết bị chạy AI (chạy 1 lần ở đầu)
#     device = get_device()

#     # Kiểm tra xem tệp video đầu vào có tồn tại không
#     if not os.path.exists(VIDEO_INPUT_PATH):
#         print(f"❌ LỖI: Không tìm thấy tệp video đầu vào tại:")
#         print(f"   {VIDEO_INPUT_PATH}")
#         print("   Hãy đảm bảo bạn đã đặt video vào thư mục 'source' và đặt tên là 'input_video.mp4'")
#         sys.exit(1) # Thoát chương trình với mã lỗi
        
#     # Bước 1 + 2: Tách âm thanh
#     extracted_audio_file = extract_audio(VIDEO_INPUT_PATH, AUDIO_OUTPUT_PATH)
    
#     if extracted_audio_file is None:
#         print("Dừng chương trình do lỗi ở Bước 1.")
#         sys.exit(1)
        
#     # Bước 3: Phiên âm (Audio to Text)
#     if os.path.exists(TRANSCRIPT_OUTPUT_PATH):
#         print(f"\nĐã tìm thấy tệp phiên âm: {TRANSCRIPT_OUTPUT_PATH}. Bỏ qua Bước 3.")
#         with open(TRANSCRIPT_OUTPUT_PATH, 'r', encoding='utf-8') as f:
#             segments = json.load(f)
#     else:
#         segments = transcribe_audio(AUDIO_OUTPUT_PATH, WHISPER_MODEL_NAME, device)
#         if segments is None: sys.exit(1)
        
#         # Lưu tệp JSON
#         print(f"\nĐang lưu kết quả phiên âm vào '{TRANSCRIPT_OUTPUT_PATH}'...")
#         try:
#             with open(TRANSCRIPT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
#                 json.dump(segments, f, indent=4, ensure_ascii=False)
#             print("✅ Đã lưu phiên âm thành công.")
#         except Exception as e:
#             print(f"❌ LỖI: Không thể lưu tệp JSON phiên âm: {e}")
#             sys.exit(1)


#     # In ra 3 segment đầu tiên để kiểm tra
#     print("\n--- Kết quả phiên âm (3 segment đầu tiên) ---")
#     for i, segment in enumerate(segments[:3]):
#         start = segment['start']
#         end = segment['end']
#         text = segment['text'].strip()
#         print(f"[{start:.2f}s -> {end:.2f}s] {text}")
#     print("---------------------------------------------")

#     # Bước 4: Dịch thuật
#     translated_segments = translate_segments(segments, TRANSLATION_MODEL_NAME, device)
#     if translated_segments is None: sys.exit(1)
        
#     # Lưu tệp JSON đã dịch
#     # print(f"\nĐang lưu kết quả dịch thuật vào '{TRANSLATED_TRANSCRIPT_PATH}'...")
#     try:
#         with open(TRANSLATED_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
#             # ensure_ascii=False RẤT QUAN TRỌNG để lưu tiếng Việt
#             json.dump(translated_segments, f, indent=4, ensure_ascii=False)
#         # print("✅ Đã lưu dịch thuật thành công.")
#     except Exception as e:
#         print(f"❌ LỖI: Không thể lưu tệp JSON dịch thuật: {e}")
#         sys.exit(1)

#     # In ra 3 segment đã dịch đầu tiên để kiểm tra
#     print("\n--- Kết quả dịch thuật (3 segment đầu tiên) ---")
#     for i, segment in enumerate(translated_segments[:3]):
#         start = segment['start']
#         end = segment['end']
#         text = segment['text'].strip()
#         print(f"[{start:.2f}s -> {end:.2f}s] {text}")
#     print("-------------------------------------------------")
    
#     # Bước 5: Chuẩn bị data cho Colab chạy
#     if not os.path.exists(TTS_DATA_PATH):
#         segments_with_audio_path = generate_tts_data_file(translated_segments, TTS_DATA_PATH)
#         if segments_with_audio_path is None: sys.exit(1)
        
#         # Cập nhật lại tệp JSON với 'audio_path'
#         try:
#             with open(TRANSLATED_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
#                 json.dump(segments_with_audio_path, f, indent=4, ensure_ascii=False)
#             print(f"✅ Đã cập nhật tệp dịch thuật với đường dẫn âm thanh.")
#             translated_segments = segments_with_audio_path # Đảm bảo chúng ta có bản mới nhất
#         except Exception as e:
#             print(f"❌ LỖI: Không thể cập nhật tệp JSON dịch thuật: {e}")
#     else:
#         print(f"\nĐã tìm thấy tệp dữ liệu TTS: {TTS_DATA_PATH}. Bỏ qua Bước 5.")
#         # Đảm bảo `translated_segments` đã có 'audio_path'
#         if 'audio_path' not in translated_segments[0]:
#             print("   Cập nhật lại đường dẫn audio cho Bước 6...")
#             for segment in translated_segments:
#                  segment_id = segment['id']
#                  segment['audio_path'] = os.path.join(SOURCE_FOLDER, f"audio_VN/{segment_id}.wav")

#     # Bước 6.1: Đồng bộ và Nối âm thanh
#     final_audio_file = synchronize_and_combine(translated_segments, FINAL_AUDIO_PATH)
#     if final_audio_file is None:
#         print("Dừng chương trình do lỗi ở Bước 6.1.")
#         sys.exit(1)

#     # Bước 6.2: Ghép âm thanh vào video
#     final_video_file = merge_audio_to_video(VIDEO_INPUT_PATH, final_audio_file, FINAL_VIDEO_PATH)
#     if final_video_file is None:
#         print("Dừng chương trình do lỗi ở Bước 6.2.")
#         sys.exit(1)

#     print("\n--- 🎉🎉🎉 HOÀN THÀNH TOÀN BỘ DỰ ÁN! 🎉🎉🎉 ---")
#     print(f"Video lồng tiếng cuối cùng của bạn đã sẵn sàng tại:")
#     print(f"{FINAL_VIDEO_PATH}")
#     print("-------------------------------------------------")

# ---- HÀM MAIN (ĐÃ CẤU TRÚC LẠI) ----
def main(args): # MỚI: 'args' được truyền vào
    device = get_device()

    # ========== CHẾ ĐỘ 1: CHUẨN BỊ (PREP) ==========
    if args.step == 'prep':
        print("--- Chạy chế độ 'PREP' (Bước 1-5) ---")
        
        # --- Bước 1 + 2: Tách âm thanh ---
        if not os.path.exists(AUDIO_OUTPUT_PATH):
            extracted_audio_file = extract_audio(VIDEO_INPUT_PATH, AUDIO_OUTPUT_PATH)
            if extracted_audio_file is None: sys.exit(1)
        else:
            print(f"Đã tìm thấy âm thanh gốc: {AUDIO_OUTPUT_PATH}. Bỏ qua Bước 1.")
        
        # --- Bước 3: Phiên âm ---
        if os.path.exists(TRANSCRIPT_OUTPUT_PATH):
            print(f"\nĐã tìm thấy tệp phiên âm: {TRANSCRIPT_OUTPUT_PATH}. Bỏ qua Bước 3.")
            with open(TRANSCRIPT_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        else:
            print("\nBắt đầu Bước 3: Phiên âm âm thanh gốc thành văn bản")
            segments = transcribe_audio(AUDIO_OUTPUT_PATH, WHISPER_MODEL_NAME, device)
            if segments is None: sys.exit(1)
            # ... (lưu tệp json)
            try:
                with open(TRANSCRIPT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(segments, f, indent=4, ensure_ascii=False)
                print(f"✅ Đã lưu phiên âm vào: {TRANSCRIPT_OUTPUT_PATH}")
            except Exception as e:
                print(f"❌ LỖI: Không thể lưu tệp JSON phiên âm: {e}")
                sys.exit(1)

        # --- Bước 4: Dịch thuật ---
        if os.path.exists(TRANSLATED_TRANSCRIPT_PATH):
            print(f"\nĐã tìm thấy tệp dịch thuật: {TRANSLATED_TRANSCRIPT_PATH}. Bỏ qua Bước 4.")
            with open(TRANSLATED_TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
                translated_segments = json.load(f)
        else:
            print("\nBắt đầu Bước 4: Dịch thuật từ văn bản sang văn bản")
            translated_segments = translate_segments(segments, TRANSLATION_MODEL_NAME, device)
            if translated_segments is None: sys.exit(1)
            # ... (lưu tệp json)
            try:
                with open(TRANSLATED_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(translated_segments, f, indent=4, ensure_ascii=False)
                print(f"✅ Đã lưu dịch thuật vào: {TRANSLATED_TRANSCRIPT_PATH}")
            except Exception as e:
                print(f"❌ LỖI: Không thể lưu tệp JSON dịch thuật: {e}")
                sys.exit(1)

        # --- Bước 5: Ghi tệp dữ liệu TTS ---
        print("\nBắt đầu Bước 5: Ghi tệp dữ liệu...")
        segments_with_audio_path = generate_tts_data_file(translated_segments, TTS_DATA_PATH)
        if segments_with_audio_path is None: sys.exit(1)

        # Cập nhật lại tệp JSON với đường dẫn âm thanh
        try:
            with open(TRANSLATED_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
                json.dump(segments_with_audio_path, f, indent=4, ensure_ascii=False)
            print(f"✅ Đã cập nhật tệp dịch thuật với đường dẫn âm thanh (dự kiến).")
        except Exception as e:
            print(f"❌ LỖI: Không thể cập nhật tệp JSON dịch thuật: {e}")

        print("\n--- ✅ Hoàn thành 'PREP' ---")
        print(f"Đã tạo tệp dữ liệu TTS tại: {TTS_DATA_PATH}")
        print("Bây giờ bạn có thể tạo các tệp .wav trong 'source/audio_VN' trước khi chạy bước 'combine'.")

    # ========== CHẾ ĐỘ 2: KẾT HỢP (COMBINE) ==========
    elif args.step == 'combine':
        print("--- Chạy chế độ 'COMBINE' (Bước 6) ---")
        
        # --- Bước 6: Đồng bộ và Ghép ---
        # Tải tệp JSON đã dịch (phải chứa 'audio_path')
        if not os.path.exists(TRANSLATED_TRANSCRIPT_PATH):
            print(f"❌ LỖI: Không tìm thấy tệp {TRANSLATED_TRANSCRIPT_PATH}.")
            print("Bạn phải chạy bước 'prep' trước.")
            sys.exit(1)
            
        print(f"Đang tải tệp dịch thuật: {TRANSLATED_TRANSCRIPT_PATH}...")
        with open(TRANSLATED_TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
            translated_segments = json.load(f)

        # Kiểm tra xem các tệp audio có thực sự tồn tại không
        first_audio_path = translated_segments[0].get('audio_path')
        if first_audio_path is None or not os.path.exists(first_audio_path):
             print(f"❌ LỖI: Không tìm thấy tệp âm thanh đầu tiên ({first_audio_path}).")
             print("Bạn đã tạo các tệp .wav trong 'source/audio_VN' chưa?")
             sys.exit(1)

        # Bước 6.1: Đồng bộ và Nối âm thanh
        final_audio_file = synchronize_and_combine(translated_segments, FINAL_AUDIO_PATH)
        if final_audio_file is None:
            print("Dừng chương trình do lỗi ở Bước 6.1.")
            sys.exit(1)

        # Bước 6.2: Ghép âm thanh vào video
        final_video_file = merge_audio_to_video(VIDEO_INPUT_PATH, final_audio_file, FINAL_VIDEO_PATH)
        if final_video_file is None:
            print("Dừng chương trình do lỗi ở Bước 6.2.")
            sys.exit(1)

        print("\n--- 🎉🎉🎉 HOÀN THÀNH TOÀN BỘ DỰ ÁN! 🎉🎉🎉 ---")
        print(f"Video lồng tiếng cuối cùng của bạn đã sẵn sàng tại:")
        print(f"{FINAL_VIDEO_PATH}")
        print("-------------------------------------------------")


if __name__ == "__main__":
    # ---- MỚI: THIẾT LẬP ARGPARSE ----
    parser = argparse.ArgumentParser(description="Quy trình lồng tiếng AI.")
    parser.add_argument(
        '--step', 
        type=str, 
        choices=['prep', 'combine'], 
        required=True, 
        help="Chọn bước để chạy: 'prep' (Bước 1-5) hoặc 'combine' (Bước 6)"
    )
    args = parser.parse_args()
    
    main(args) # Chạy hàm main với các đối số




