import torch
import fasttext
import numpy as np
# Load mô hình FastText
fasttext_model = fasttext.load_model("fasttext_vietnamese.bin")

# Hàm lấy embedding từ FastText
def get_fasttext_embedding(sentence):
    words = sentence.split()  # Chia câu thành từ
    embeddings = [fasttext_model.get_word_vector(word) for word in words]
    return torch.from_numpy(np.array(embeddings))  # Tensor shape: (số từ, 300)