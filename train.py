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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def guided_attention_loss(attn_weights, input_lengths, output_lengths, g=0.2):
    B, T_dec, T_enc = attn_weights.size()
    loss = 0.0

    for b in range(B):
        N = int(input_lengths[b])
        M = int(output_lengths[b])

        grid_i = torch.arange(M).unsqueeze(1).to(attn_weights.device) / M
        grid_j = torch.arange(N).unsqueeze(0).to(attn_weights.device) / N

        W = 1.0 - torch.exp(-((grid_i - grid_j) ** 2) / (2 * g * g))
        A = attn_weights[b, :M, :N]
        loss += torch.mean(A * W)

    return loss / B


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
        # Giả sử mỗi mel sequence có cùng độ dài (do collate pad), dùng làm output_lengths
        mel_before, mel_after, stop_logits, alignments = model(
            fasttext_embeddings, lengths, target_mel=mel)
        if mel_before.size(1) < mel.size(1):
            mel_before = mel_before.permute(0, 2, 1)
            mel_before = F.interpolate(mel_before, size=(
                mel.size(1)), mode='linear', align_corners=False)
            mel_before = mel_before.permute(0, 2, 1)
        else:
            mel_before = mel_before[:, :mel.size(1), :]

        if mel_after.size(1) < mel.size(1):
            mel_after = mel_after.permute(0, 2, 1)
            mel_after = F.interpolate(mel_after, size=(
                mel.size(1)), mode='linear', align_corners=False)
            mel_after = mel_after.permute(0, 2, 1)
        else:
            mel_after = mel_after[:, :mel.size(1), :]

        stop_logits = stop_logits[:, :mel.size(1), :]
        output_lengths = torch.tensor(
            [alignments.shape[1]] * alignments.size(0)).to(device)
        # Mel losses
        loss_mel_before = criterion(mel_before, mel)
        loss_mel_after = criterion(mel_after, mel)

        # Stop token loss (dự đoán 1 tại frame cuối)
        stop_targets = torch.zeros_like(stop_logits).to(device)
        stop_targets[:, -1, 0] = 1.0
        loss_stop = F.binary_cross_entropy_with_logits(
            stop_logits, stop_targets)

        # Guided attention loss
        loss_attn = guided_attention_loss(
            alignments, input_lengths=lengths, output_lengths=output_lengths
        )
        loss = loss_mel_before + loss_mel_after + 0.5 * loss_stop + 1.0 * loss_attn
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Log every 10th step (real-time log output)
        if batch_idx % 10 == 0:
            logging.info(
                f"Epoch [{epoch+1}], Step [{batch_idx+1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, mel_before: {loss_mel_before.item():.4f}, "
                f"mel_after: {loss_mel_after.item():.4f}, stop: {loss_stop.item():.4f}, "
                f"attn: {loss_attn.item():.4f}"
            )

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

num_epochs = 5
logging.info("Start training")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, epoch)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

save_model(model)
