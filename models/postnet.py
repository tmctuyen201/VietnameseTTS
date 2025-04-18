import torch.nn as nn


class Postnet(nn.Module):
    def __init__(self, mel_dim, postnet_dim, num_convs=5, dropout=0.1):
        super(Postnet, self).__init__()
        layers = []
        for i in range(num_convs - 1):
            layers.append(nn.Conv1d(mel_dim if i == 0 else postnet_dim,
                          postnet_dim, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(postnet_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
        # Lớp cuối dùng để chuyển sang kênh mel_dim
        layers.append(nn.Conv1d(postnet_dim, mel_dim,
                      kernel_size=5, padding=2))
        self.convs = nn.Sequential(*layers)

    def forward(self, mel):
        # mel: (B, T, mel_dim) --> chuyển thành (B, mel_dim, T) cho conv1d
        mel = mel.transpose(1, 2)
        out = self.convs(mel)
        out = out.transpose(1, 2)
        return out
