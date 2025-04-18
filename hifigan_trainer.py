import os
from datasets import load_dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch

# Tải dataset từ Hugging Face
dataset_name = "trinhtuyen201/my-audio-dataset"  # Tên bộ dữ liệu của bạn
dataset = load_dataset(dataset_name, split="train")
# Thiết lập Mel-spectrogram extractor
mel_transform = T.MelSpectrogram(
    sample_rate=22050, n_mels=80, hop_length=256, n_fft=1024)

# Hàm trích xuất Mel-spectrogram từ âm thanh


def extract_mel(waveform):
    mel_spectrogram = mel_transform(waveform)
    return mel_spectrogram.float()


# # Giả sử dataset chứa các âm thanh dạng .wav, ta sẽ áp dụng hàm trên cho mỗi file âm thanh trong dataset
# mel_spectrograms = []
# for sample in dataset:
#     waveform = sample["audio"]["array"]  # Dữ liệu âm thanh
#     mel_spectrogram = extract_mel(waveform)
#     mel_spectrograms.append(mel_spectrogram)

# # Convert to numpy arrays hoặc tensor
# mel_spectrograms = torch.stack(mel_spectrograms)  # hoặc numpy array nếu cần

# Tạo thư mục lưu Mel-spectrogram và audio nếu chưa có
os.makedirs("data/mels", exist_ok=True)
os.makedirs("data/wavs", exist_ok=True)

# Giả sử mỗi sample trong dataset có cột `audio` chứa file path âm thanh và cột `text` chứa văn bản
for i, sample in enumerate(dataset):
    # Truyền mảng âm thanh từ dataset (sample["audio"]["array"])
    # Đây là mảng âm thanh đã có sẵn (numpy array hoặc list)
    waveform = sample["audio"]["array"]

    # Chuyển waveform thành tensor nếu nó là numpy array
    if isinstance(waveform, np.ndarray):
        waveform = torch.tensor(waveform).float()
    if waveform.ndimension() == 1:
        waveform = waveform.unsqueeze(0)  # Thêm chiều kênh vào (1, T)
    # Trích xuất Mel-spectrogram
    mel_spectrogram = extract_mel(waveform)

    # Lưu Mel-spectrogram vào file .npy
    mel_file_path = f"data/mels/sample_{i}.npy"
    np.save(mel_file_path, mel_spectrogram.numpy())

    # Lưu audio vào thư mục wavs (đảm bảo rằng waveform là tensor)
    wav_file_path = f"data/wavs/sample_{i}.wav"
    # Sử dụng `waveform` đã có sẵn
    torchaudio.save(wav_file_path, waveform, 22050)
