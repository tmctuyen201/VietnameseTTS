import torch
from torch.utils.data import Dataset
import numpy as np
from utils import get_fasttext_embedding
import torchaudio.transforms as T

mel_transform = T.MelSpectrogram(
    sample_rate=22050, n_mels=80, hop_length=256, n_fft=1024)


class TTS_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # Chuyển đổi audio và transcription thành Mel-spectrogram và phoneme
        mel = self.process_audio(example)  # Chuyển audio thành Mel-spectrogram
        # Chuyển transcription thành phonemes
        text = self.process_text(example)
        fasttext_embeddings = get_fasttext_embedding(text)
        # Bạn cũng có thể lưu trữ độ dài của phonemes để xử lý padding
        length = len(fasttext_embeddings)
        return fasttext_embeddings, mel, length

    # Ví dụ chuyển văn bản thành phoneme
    def process_text(self, example):
        text = example["transcription"]
        # phonemes = text_to_phonemes(text)
        return text
    # Hàm chuyển đổi từ audio sang Mel-spectrogram

    def extract_mel(self, waveform):
        mel_spectrogram = mel_transform(waveform)
        return mel_spectrogram.float()

    # Ví dụ xử lý một mẫu audio
    def process_audio(self, example):
        # Đảm bảo lấy đường dẫn đúng từ dataset
        waveform = example["audio"]["array"]
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform).float()
        if waveform.ndimension() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.extract_mel(waveform)
        mel = mel.squeeze(0)
        return mel
