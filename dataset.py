from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
import numpy as np
from utils import get_fasttext_embedding

class TTS_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # Chuyển đổi audio và transcription thành Mel-spectrogram và phoneme
        mel = self.process_audio(example)  # Chuyển audio thành Mel-spectrogram
        text = self.process_text(example)  # Chuyển transcription thành phonemes
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
    def audio_to_mel_spectrogram(self, audio, sr=16000, n_mels=80, hop_length=512, n_fft=2048, fmin=0, fmax=8000):
        # Tính Mel-spectrogram từ tín hiệu âm thanh
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmin=fmin, fmax=fmax)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Chuyển đổi sang decibel
        return mel_spectrogram

    # Ví dụ xử lý một mẫu audio
    def process_audio(self, example):
        audio = example["audio"]["array"]  # Đảm bảo lấy đường dẫn đúng từ dataset
        mel = self.audio_to_mel_spectrogram(audio)
        return mel