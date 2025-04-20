# synthesis.py

import argparse
import json
from config.config import Config
from models.mel2wav import generate_audio, save_waveform
from models.text2mel import generate_mel
import torch
from model import HybridTTS  # Giả sử bạn đã có mô hình Hybrid TTS
from models.hifigan import Generator
import fasttext

# Định nghĩa hàm để tải các mô hình


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_models(device):
    # Load mô hình FastText
    fasttext_model = fasttext.load_model(
        './output_model/fasttext_vietnamese_1.bin')
    config = Config()
    # Load mô hình Hybrid TTS
    hybrid_tts_model = HybridTTS(config)  # Load model Hybrid TTS của bạn
    hybrid_tts_model.to(device)
    # Load trọng số từ checkpoint
    checkpoint = torch.load(
        './hybrid_tts_model.pth', map_location=device)
    hybrid_tts_model.load_state_dict(checkpoint, strict=False)
    hybrid_tts_model.eval()  # Chuyển mô hình sang chế độ inference
    config_file = "assets/hifigan/config.json"
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    # Load mô hình HiFi-GAN Generator
    generator = Generator(h)  # Load HiFi-GAN generator model
    state_dict_g = torch.load(
        './output_model/g_00035000', map_location=device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.to(device)
    generator.eval()  # Chuyển mô hình HiFi-GAN vào chế độ inference

    return fasttext_model, hybrid_tts_model, generator

# Main function để thực hiện pipeline


def main():
    # Parse đối số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Synthesis audio from text")
    parser.add_argument("text", type=str, help="Input text to generate speech")
    parser.add_argument("output_path", type=str,
                        help="Path to save the output audio")

    args = parser.parse_args()

    # Chọn device (GPU nếu có, nếu không thì CPU)
    device = torch.device("cpu")

    # Load các mô hình cần thiết
    fasttext_model, hybrid_tts_model, generator = load_models(device)

    # Tạo Mel-spectrogram từ text
    mel_spectrogram = generate_mel(
        args.text, hybrid_tts_model, fasttext_model, device)

    # Tạo âm thanh từ Mel-spectrogram
    waveform = generate_audio(mel_spectrogram, generator, device)

    # Lưu âm thanh vào file .wav
    save_waveform(waveform, args.output_path)
    print(f"Audio saved as {args.output_path}")


if __name__ == "__main__":
    main()
