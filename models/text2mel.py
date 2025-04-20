# text2mel.py

import torch
import fasttext
import numpy as np
from utils import get_fasttext_embedding, preprocess_text

# Load mô hình FastText để chuyển text thành vector


def load_fasttext_model(fasttext_model_path):
    model = fasttext.load_model(fasttext_model_path)
    return model

# Chuyển văn bản thành vector bằng FastText


def text_to_vector(text, fasttext_model):
    vector = get_fasttext_embedding(text)
    return vector, len(vector)


# Chuyển văn bản thành Mel-spectrogram bằng Hybrid TTS
def text_to_mel(text, hybrid_tts_model, fasttext_model, device):
    # Tiền xử lý văn bản
    processed_text = preprocess_text(text)

    # Chuyển văn bản thành vector (embedding)
    text_vector, lenths = text_to_vector(processed_text, fasttext_model)
    text_vector = text_vector.unsqueeze(0)
    input_lengths = torch.tensor([lenths])
    # Chuyển văn bản thành Mel-spectrogram bằng Hybrid TTS
    mel_before, mel_after, stop_logits, alignments = hybrid_tts_model(
        text_vector, input_lengths)
    return mel_after

# Để gọi từ `synthesis.py`, bạn chỉ cần gọi hàm `text_to_mel`


def generate_mel(text, hybrid_tts_model, fasttext_model, device):
    return text_to_mel(text, hybrid_tts_model, fasttext_model, device)
