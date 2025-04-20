import torch
import torchaudio
from models.hifigan import Generator

# Load HiFi-GAN Generator model


def load_hifigan_generator(checkpoint_path, device):
    generator = Generator()
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()  # Chuyển mô hình sang chế độ inference
    return generator

# Tạo âm thanh từ Mel-spectrogram bằng HiFi-GAN Generator


def mel_to_waveform(mel_spectrogram, generator, device):
    # Chuyển Mel-spectrogram thành waveform (audio)
    # Giả sử mel_spectrogram có kích thước [batch_size, n_mels, time_steps]
    mel_spectrogram = mel_spectrogram.transpose(1, 2)

    # Tạo âm thanh từ mel-spectrogram
    with torch.no_grad():
        generated_waveform = generator(mel_spectrogram)

    return generated_waveform.to(device)

# Lưu waveform thành tệp .wav


def save_waveform(waveform, output_path, sample_rate=16000):
    torchaudio.save(output_path, waveform.squeeze(0), sample_rate)


def generate_audio(mel_spectrogram, generator, device):
    return mel_to_waveform(mel_spectrogram, generator, device)
