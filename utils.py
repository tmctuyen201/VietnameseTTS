import torch
import fasttext
import numpy as np
import underthesea
import re
from vinorm import TTSnorm
from pyvi import ViTokenizer
# Load mô hình FastText
fasttext_model = fasttext.load_model("fasttext_vietnamese.bin")

# Hàm lấy embedding từ FastText


def get_fasttext_embedding(sentence):
    words = sentence.split()  # Chia câu thành từ
    embeddings = [fasttext_model.get_word_vector(word) for word in words]
    return torch.from_numpy(np.array(embeddings))  # Tensor shape: (số từ, 300)


def preprocess_text(text):
    """Tiền xử lý văn bản tiếng Việt"""
    text = TTSnorm(text, punc=False, unknown=True, lower=True,
                   rule=False)  # Chuẩn hóa tiếng Việt bằng Vinorm
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'[\d]+|[^\w\s]', '', text)
    text = ViTokenizer.tokenize(text)  # Tách từ
    return text
