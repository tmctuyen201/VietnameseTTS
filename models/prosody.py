import torch.nn as nn

class ProsodyPredictor(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super(ProsodyPredictor, self).__init__()
        # Ví dụ đơn giản dùng CNN hoặc MLP cho từng thông số
        self.duration_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.pitch_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.energy_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, enc_out):
        # enc_out: (B, T, hidden_dim)
        duration = self.duration_pred(self.dropout(enc_out)).squeeze(-1)
        pitch = self.pitch_pred(self.dropout(enc_out)).squeeze(-1)
        energy = self.energy_pred(self.dropout(enc_out)).squeeze(-1)
        # Sử dụng softplus cho duration (đảm bảo giá trị dương)
        duration = F.softplus(duration)
        return duration, pitch, energy
