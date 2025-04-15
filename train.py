from config.config import Config
from dataset import TTS_Dataset
from model import HybridTTS
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

config = Config()
model = HybridTTS(config).cuda()  # Đây là mô hình PyTorch của bạn.
# MSE Loss hoặc L1 Loss có thể được sử dụng
criterion = nn.MSELoss()

# Optimizer: Adam hoặc AdamW
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

def custom_collate_fn(batch, target_length=None):
    """
    batch: list các tuple (fasttext_embeddings, mel, length)
      - fasttext_embeddings: tensor với shape (T, fasttext_dim)
      - mel: numpy array hoặc tensor với shape (T_mel, mel_dim) 
      - length: int, số token của fasttext_embeddings
    """
    fasttext_list, mel_list, lengths = zip(*batch)
    
    # Nếu mel chưa là tensor, chuyển từ numpy sang tensor:
    mel_list = [torch.tensor(m) if not torch.is_tensor(m) else m for m in mel_list]
    
    # Nếu cần, bạn có thể transpose mel nếu mô hình của bạn yêu cầu (B, T, mel_dim):
    # Giả sử mel có shape (n_mels, T) từ librosa, bạn cần chuyển thành (T, n_mels)
    mel_list = [m.transpose(0, 1) if m.dim() == 2 and m.size(0) == 80 else m for m in mel_list]
    
    # Pad fastText embeddings cho batch: (B, max_fasttext_length, fasttext_dim)
    fasttext_padded = pad_sequence(fasttext_list, batch_first=True, padding_value=0)
    
    # Pad mel spectrograms cho batch: (B, max_T_mel, mel_dim)
    mel_padded = pad_sequence(mel_list, batch_first=True, padding_value=0)
    
    # Lấy độ dài thực tế của mỗi mẫu (số token)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    # Ensure that mel_padded và fasttext_padded có độ dài tương thích
    B, T = fasttext_padded.size(0), fasttext_padded.size(1)
    max_mel_length = mel_padded.size(1)
    
    if target_length:
        # Nếu yêu cầu, tiến hành padding/cắt Mel-spectrograms hoặc downsample
        if max_mel_length > target_length:
            # Trimming: Cắt Mel-spectrograms theo target_length
            mel_padded = mel_padded[:, :target_length, :]
        elif max_mel_length < target_length:
            # Up-sampling: Nhân Mel-spectrograms theo tỷ lệ lên target_length
            mel_padded = F.interpolate(mel_padded, size=(target_length, mel_padded.size(2)), mode='linear', align_corners=False)    
    # Kiểm tra lại chiều dài của mel_padded và fasttext_padded đã đồng bộ chưa
    return fasttext_padded, mel_padded, lengths
def save_model(model, path="hybrid_tts_model.pth"):
    """Lưu mô hình sau khi huấn luyện hoàn tất"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch_idx, (fasttext_embeddings, mel, lengths) in enumerate(train_loader):
        optimizer.zero_grad()

        # Đưa dữ liệu vào GPU nếu có
        fasttext_embeddings = fasttext_embeddings.cuda()
        mel = mel.cuda()
        lengths = lengths.cuda()
        lengths = lengths.cpu()
        # Tiến hành một bước forward
        outputs = model(fasttext_embeddings, lengths, target_mel=mel)  # Tính toán Mel-spectrogram từ mô hình

        mel_output = outputs["mel_final"]  # Mel spectrogram sau postnet

        # Tính loss giữa Mel-spectrogram dự đoán và thực tế
        loss = criterion(mel_output, mel)
        total_loss += loss.item()

        # Backpropagation và cập nhật trọng số
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)
print("Start downloading dataset")
# Tải dữ liệu từ Hugging Face
dataset_name = "trinhtuyen201/my-audio-dataset"  # Tên bộ dữ liệu của bạn
dataset = load_dataset(dataset_name, split="train")
print("Finishing downloading dataset")
target_length = 73068  # Hoặc chiều dài token của fasttext_embeddings mà bạn mong muốn

# Tạo dataset và dataloader
train_dataset = TTS_Dataset(dataset)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
# Huấn luyện qua nhiều epochs
num_epochs = 3
print("Start training")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
save_model(model)