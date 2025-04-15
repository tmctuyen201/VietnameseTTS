import argparse
import re
import fasttext
import underthesea
from vinorm import TTSnorm
import torch
import torch.nn as nn
import torch.nn.functional as F

def preprocess_text(text):
    """Tiền xử lý văn bản tiếng Việt"""
    text = TTSnorm(text, punc = False, unknown = True, lower = True, rule = False )  # Chuẩn hóa tiếng Việt bằng Vinorm
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'\d+', '', text)  # Loại bỏ số
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = underthesea.word_tokenize(text, format="text")  # Tách từ
    return text

def train_fasttext(input_file_path, output_file_path, num_epochs):
    # Đọc nội dung file
    with open(input_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tiền xử lý văn bản
    cleaned_text = preprocess_text(text)

    # Lưu văn bản đã xử lý ra file mới
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text + "\n")
    print("Lưu file thành công!")

    # Huấn luyện mô hình FastText
    model = fasttext.train_unsupervised(output_file_path, model="skipgram", dim=300, epoch=num_epochs)

    # Lưu mô hình
    model.save_model("fasttext_vietnamese.bin")
    print(f"Mô hình FastText đã được huấn luyện và lưu thành công sau {num_epochs} epochs!")

def main():
    # Tạo đối tượng parser
    parser = argparse.ArgumentParser(description='Train FastText model on Vietnamese text')

    # Thêm các argument từ dòng lệnh
    parser.add_argument('--input_path', type=str, default='input_text.txt', help='Đường dẫn đến tệp văn bản đầu vào')
    parser.add_argument('--output_path', type=str, default='cleaned_text.txt', help='Đường dẫn đến tệp văn bản đã xử lý')
    parser.add_argument('--num_epochs', type=int, default=10, help='Số epoch cho quá trình huấn luyện mô hình FastText')

    # Thêm đối số model (có thể là skipgram hoặc cbow)
    parser.add_argument('--model', type=str, default='skipgram', choices=['skipgram', 'cbow'], help='Loại mô hình FastText: skipgram hoặc cbow')

    # Lấy các đối số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm train_fasttext với các tham số từ dòng lệnh
    train_fasttext(args.input_path, args.output_path, args.num_epochs)

if __name__ == "__main__":
    main()
