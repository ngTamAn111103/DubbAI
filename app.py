import subprocess
import sys
import os
import torch
import whisper
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import json
import pprint
from transformers import MarianMTModel, MarianTokenizer,pipeline 
import math
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
AUDIO_OUTPUT_NAME = "original_audio.wav" # Bước 1 +2

VIDEO_INPUT_PATH = os.path.join(SOURCE_FOLDER, VIDEO_INPUT_NAME)
AUDIO_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, AUDIO_OUTPUT_NAME)

# Tệp JSON chứa kết quả phiên âm
TRANSCRIPT_OUTPUT_NAME = "original_transcript.json" # Bước 3
TRANSCRIPT_OUTPUT_PATH = os.path.join(SOURCE_FOLDER, TRANSCRIPT_OUTPUT_NAME)
# Tệp JSON dịch thuật Anh -> Việt
TRANSLATED_TRANSCRIPT_NAME = "translated_transcript.json" # Bước 4
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
WHISPER_MODEL_NAME = "large-v3"
# Mô hình dịch thuật
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"
# TRANSLATION_MODEL_NAME = "vinai/vinai-translate-en2vi"

# Cấu hình tùy chọn cho Whisper
# Đây là nơi bạn "tinh chỉnh" (tune) để sửa lỗi mốc thời gian
WHISPER_OPTIONS = {
    "no_speech_threshold": 0.3,  # Hạ thấp ngưỡng để dễ phát hiện im lặng hơn (Mặc định 0.6)
    "hallucination_silence_threshold": 3.0, # Xóa ảo giác trong khoảng lặng > 3 giây
    "word_timestamps": True,     # Bật để tăng độ chính xác của mốc thời gian
    "fp16": False                # Đặt là False nếu chạy trên CPU (an toàn)
}
# Cấu hình VAD (Voice Activity Detection)
VAD_OPTIONS = {
    "min_silence_len": 1000, # (ms) Khoảng lặng tối thiểu để tính là "im lặng"
    "silence_thresh": -2,   # Giá trị cao hơn: Chỉ những âm thanh thực sự lớn mới được coi là "có tiếng".
    "keep_silence": 250      # (ms) Giữ lại một chút im lặng ở đầu/cuối
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
    


def transcribe_audio(audio_path: str, model_name: str, device: str) -> list[dict] | None:
    """
    Phiên âm bằng VAD + Whisper để có mốc thời gian chính xác.
    """
    # print(f"\nBắt đầu Bước 3: Phiên âm (Sử dụng VAD)...")
    try:
        # 1. Tải mô hình Whisper
        # print(f"Đang tải mô hình Whisper '{model_name}'...")
        model = whisper.load_model(model_name, device=device)
        # print("Tải mô hình hoàn tất.")

        # Cập nhật tùy chọn fp16
        transcribe_options = WHISPER_OPTIONS.copy()
        transcribe_options["fp16"] = (device != "cpu")
        # print(f"Đang phiên âm với các tùy chọn: {transcribe_options}")

        # 2. Tải âm thanh bằng Pydub
        # print(f"Đang tải âm thanh từ: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)

        # 3. Chạy VAD (Phát hiện các đoạn không im lặng)
        # print(f"Đang chạy VAD (Phát hiện giọng nói)...")
        speech_chunks = detect_nonsilent(
            audio,
            min_silence_len=VAD_OPTIONS["min_silence_len"],
            silence_thresh=VAD_OPTIONS["silence_thresh"]
        )
        
        if not speech_chunks:
            print("❌ LỖI: VAD không tìm thấy bất kỳ giọng nói nào trong tệp.")
            return None

        total_chunks = len(speech_chunks)
        print(f"VAD đã tìm thấy {total_chunks} đoạn có giọng nói.")
        
        all_segments = []
        segment_id_counter = 0
        temp_chunk_path = os.path.join(SOURCE_FOLDER, "temp_chunk.wav") # Định nghĩa 1 lần

        # 4. Lặp qua từng đoạn có tiếng và chạy Whisper
        for i, chunk_ms in enumerate(speech_chunks):
            original_start_ms, original_end_ms = chunk_ms
            
            # === LOG MỚI ===
            print(f"\n   --- VAD Chunk {i+1}/{total_chunks} ---")
            print(f"   Đoạn VAD gốc: {original_start_ms/1000:.2f}s -> {original_end_ms/1000:.2f}s")
            
            # Giữ lại một chút đệm im lặng (tùy chọn)
            start_ms = max(0, original_start_ms - VAD_OPTIONS["keep_silence"])
            end_ms = min(len(audio), original_end_ms + VAD_OPTIONS["keep_silence"])
            
            # === LOG MỚI ===
            print(f"   Đoạn đã đệm (gửi cho Whisper): {start_ms/1000:.2f}s -> {end_ms/1000:.2f}s (Thời lượng: {(end_ms-start_ms)/1000:.2f}s)")
            
            # Cắt đoạn âm thanh
            audio_chunk = audio[start_ms:end_ms]
            
            # Cần lưu ra tệp tạm để Whisper đọc
            audio_chunk.export(temp_chunk_path, format="wav")

            # 5. Chạy Whisper trên đoạn âm thanh đã cắt
            # === LOG MỚI ===
            print(f"   ...Đang chạy Whisper trên đoạn này...")
            result = model.transcribe(temp_chunk_path, task="transcribe", **transcribe_options)
            
            if not result['segments']:
                # === LOG MỚI ===
                print(f"   ...Whisper không tìm thấy văn bản nào trong đoạn này.")
                continue # Bỏ qua nếu Whisper không nghe thấy gì

            # === LOG MỚI ===
            print(f"   ...Whisper tìm thấy {len(result['segments'])} segment(s) trong đoạn này:")

            # 6. Điều chỉnh lại mốc thời gian
            for segment in result['segments']:
                # Tính toán mốc thời gian cuối cùng bằng cách cộng offset
                offset_start_sec = (segment['start'] * 1000 + start_ms) / 1000.0
                offset_end_sec = (segment['end'] * 1000 + start_ms) / 1000.0
                
                new_segment = {
                    'id': segment_id_counter,
                    'start': offset_start_sec,
                    'end': offset_end_sec,
                    'text': segment['text']
                }
                
                # === LOG MỚI ===
                text_preview = segment['text'].strip()[:50] # Lấy 50 ký tự đầu
                if len(segment['text'].strip()) > 50:
                    text_preview += "..."
                print(f"      -> Segment ID {segment_id_counter}: [{offset_start_sec:.2f}s -> {offset_end_sec:.2f}s] {text_preview}")

                all_segments.append(new_segment)
                segment_id_counter += 1
        
        # Dọn dẹp tệp tạm
        if os.path.exists(temp_chunk_path):
            os.remove(temp_chunk_path)

        print(f"✅ Bước 3 hoàn thành! Đã phiên âm {len(all_segments)} segments.")
        
        if all_segments:
            seg0 = all_segments[0]
            print(f"   Kiểm tra: Segment 0 (ID {seg0['id']}) bắt đầu từ {seg0['start']:.2f}s")

        return all_segments

    except Exception as e:
        print(f"❌ LỖI trong quá trình phiên âm VAD: {e}")
        return None
        
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
    



def main():
    device = get_device()

    # ========== CHẾ ĐỘ 1: CHUẨN BỊ (PREP) ==========  
    # --- Bước 3: Phiên âm ---
    if os.path.exists(TRANSCRIPT_OUTPUT_PATH):
        with open(TRANSCRIPT_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            segments = json.load(f)

    # --- Bước 4: Dịch thuật ---
    if os.path.exists(TRANSLATED_TRANSCRIPT_PATH):
        print(f"\nĐã tìm thấy tệp dịch thuật: {TRANSLATED_TRANSCRIPT_PATH}. Bỏ qua Bước 4.")
        with open(TRANSLATED_TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
            translated_data = json.load(f) # Load dữ liệu đã dịch
    else:
        # Truyền TOÀN BỘ dict (segments) vào
        translated_data = translate_segments(segments, TRANSLATION_MODEL_NAME, device)
        
        if translated_data is None:
            print("Dịch thuật thất bại.")
            return
            
        # Lưu TỆP DICT ĐÃ DỊCH
        try:
            with open(TRANSLATED_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Đã lưu dịch thuật vào: {TRANSLATED_TRANSCRIPT_PATH}")
        except Exception as e:
            print(f"❌ LỖI: Không thể lưu tệp JSON dịch thuật: {e}")
            sys.exit(1)

    # --- Bước 5: Ghi tệp dữ liệu TTS ---
    print("\nBắt đầu Bước 5: Ghi tệp dữ liệu...")
    segments_with_audio_path = generate_tts_data_file(translated_data, TTS_DATA_PATH)
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






main()