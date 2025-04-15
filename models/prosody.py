import torch.nn as nn


class ProsodyPredictor(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super(ProsodyPredictor, self).__init__()
        # Các lớp Linear để dự đoán duration, pitch, và energy
        self.duration_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output shape: (B, T, 1)
        )
        self.pitch_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output shape: (B, T, 1)
        )
        self.energy_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output shape: (B, T, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out):
        # Làm phẳng tensor (B, T, hidden_dim) thành (B * T, hidden_dim) để đưa vào Linear layer
        B, T, H = enc_out.size()  # B = batch size, T = sequence length, H = hidden_dim
        assert enc_out.size(
            2) == H, f"Expected hidden_dim to be {H}, but got {enc_out.size(2)}"
        # Làm phẳng thành (B * T, hidden_dim)
        enc_out_flat = enc_out.view(-1, H)
        # Truyền qua các lớp dự đoán
        duration = self.duration_pred(self.dropout(
            enc_out_flat)).view(B, T, 1)  # (B, T, 1)
        pitch = self.pitch_pred(self.dropout(
            enc_out_flat)).view(B, T, 1)  # (B, T, 1)
        energy = self.energy_pred(self.dropout(
            enc_out_flat)).view(B, T, 1)  # (B, T, 1)

        # Trả về các giá trị dự đoán
        return duration, pitch, energy
