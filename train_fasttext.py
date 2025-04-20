import argparse
import fasttext
from utils import preprocess_text


def train_fasttext(input_files, output_file_path, num_epochs, model_type):
    # Gộp nội dung từ nhiều file
    raw_text = ""
    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text += f.read() + "\n"

    # Tiền xử lý
    cleaned_text = preprocess_text(raw_text)

    # Lưu văn bản đã xử lý ra file mới
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text + "\n")
    print("✅ Đã lưu văn bản đã xử lý vào:", output_file_path)

    # Huấn luyện mô hình FastText
    model = fasttext.train_unsupervised(
        output_file_path, model=model_type, dim=300, epoch=num_epochs)

    model.save_model("fasttext_vietnamese.bin")
    print(
        f"✅ Đã huấn luyện và lưu mô hình FastText ({model_type}) thành công!")


def main():
    parser = argparse.ArgumentParser(
        description='Train FastText model on Vietnamese text')
    parser.add_argument('--input_paths', nargs='+', required=True,
                        help='Danh sách các file văn bản cần gộp để huấn luyện')
    parser.add_argument('--output_path', type=str, default='cleaned_text.txt')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--model', type=str,
                        default='skipgram', choices=['skipgram', 'cbow'])

    args = parser.parse_args()

    train_fasttext(args.input_paths, args.output_path,
                   args.num_epochs, args.model)


if __name__ == "__main__":
    main()
