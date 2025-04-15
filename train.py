import logging
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from config.config import Config
from dataset import TTS_Dataset
from model import HybridTTS
from datasets import load_dataset


# Cấu hình logging để hiển thị thông tin ra terminal và vào file
logging.basicConfig(
    level=logging.INFO,  # Ghi lại tất cả các log ở mức INFO trở lên
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Hiển thị log trên terminal (real-time)
        # Lưu log vào file training_log.txt (tuỳ chọn)
        logging.FileHandler('training_log.txt', mode='w')
    ]
)

# Khởi tạo mô hình và các thành phần
config = Config()
model = HybridTTS(config).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4,
                       betas=(0.9, 0.98), eps=1e-9)


def custom_collate_fn(batch, target_length=None):
    fasttext_list, mel_list, lengths = zip(*batch)

    mel_list = [torch.tensor(m) if not torch.is_tensor(m)
                else m for m in mel_list]
    mel_list = [m.transpose(0, 1) if m.dim() == 2 and m.size(
        0) == 80 else m for m in mel_list]

    fasttext_padded = pad_sequence(
        fasttext_list, batch_first=True, padding_value=0)
    mel_padded = pad_sequence(mel_list, batch_first=True, padding_value=0)

    lengths = torch.tensor(lengths, dtype=torch.long)

    B, T = fasttext_padded.size(0), fasttext_padded.size(1)
    max_mel_length = mel_padded.size(1)

    if target_length:
        if max_mel_length > target_length:
            mel_padded = mel_padded[:, :target_length, :]
        elif max_mel_length < target_length:
            mel_padded = F.interpolate(mel_padded, size=(
                target_length, mel_padded.size(2)), mode='linear', align_corners=False)

    return fasttext_padded, mel_padded, lengths


def save_model(model, path="hybrid_tts_model.pth"):
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (fasttext_embeddings, mel, lengths) in enumerate(train_loader):
        optimizer.zero_grad()

        fasttext_embeddings = fasttext_embeddings.cuda()
        mel = mel.cuda().float()
        lengths = lengths.cpu().float()

        outputs = model(fasttext_embeddings, lengths, target_mel=mel)

        if outputs.size(1) < mel.size(1):
            outputs = outputs.permute(0, 2, 1)
            outputs = F.interpolate(outputs, size=(
                mel.size(1)), mode='linear', align_corners=False)
            outputs = outputs.permute(0, 2, 1)
        else:
            outputs = outputs[:, :mel.size(1), :]

        loss = criterion(outputs, mel)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Log every 10th step (real-time log output)
        if batch_idx % 10 == 0:
            logging.info(
                f"Epoch [{epoch+1}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch [{epoch+1}], Average Loss: {avg_loss:.4f}")
    return avg_loss


# Start training
logging.info("Start downloading dataset")
dataset_name = "trinhtuyen201/my-audio-dataset"
dataset = load_dataset(dataset_name, split="train")
logging.info("Finishing downloading dataset")
target_length = 73068

train_dataset = TTS_Dataset(dataset)
train_loader = DataLoader(train_dataset, batch_size=16,
                          shuffle=True, collate_fn=custom_collate_fn)

num_epochs = 3
logging.info("Start training")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, epoch)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

save_model(model)
