import torch.nn as nn
import torch
class Tacotron2Decoder(nn.Module):
    def __init__(self, hidden_dim, mel_dim, num_layers=2, dropout=0.1):
        super(Tacotron2Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim
        # Prenet để xử lý đầu vào (mel frame trước đó)
        self.prenet = nn.Sequential(
            nn.Linear(mel_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Decoder RNN có thể là LSTM nhiều lớp
        self.decoder_rnn = nn.LSTM(hidden_dim + hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        # Projection sang mel-spectrogram
        self.mel_proj = nn.Linear(hidden_dim, mel_dim)
    
    def forward(self, encoder_memory, target_mel=None, pitch=None, energy=None):
        # Nếu đang huấn luyện, dùng teacher forcing với target_mel
        # encoder_memory: (B, T_enc, hidden_dim)
        # target_mel: (B, T_mel, mel_dim)
        # Ở chế độ inference: seed với vector mel frame đầu tiên (ví dụ zeros)
        if pitch is not None and energy is not None:
            # Giả sử pitch và energy có shape (B, T)
            pitch_energy = torch.cat([pitch.unsqueeze(-1), energy.unsqueeze(-1)], dim=-1)  # (B, T, 2)
            # Thêm pitch và energy vào encoder_memory
            encoder_memory = torch.cat([encoder_memory, pitch_energy], dim=-1)
        B, T, _ = encoder_memory.size()
        if target_mel is not None:
            outputs = []
            prev_mel = torch.zeros(B, self.mel_dim).to(encoder_memory.device)
            
            for t in range(T):
                prenet_out = self.prenet(prev_mel).unsqueeze(1)
                
                # Attention: query = prenet_out, keys & values = encoder_memory
                attn_output, _ = self.attention(prenet_out, encoder_memory, encoder_memory)
                
                # Concatenate prenet_out và attn_output
                decoder_input = torch.cat([prenet_out, attn_output], dim=-1)
                
                out, _ = self.decoder_rnn(decoder_input)
                mel_frame = self.mel_proj(out.squeeze(1))
                outputs.append(mel_frame.unsqueeze(1))
                
                # Teacher forcing: dùng target mel của bước hiện tại
                prev_mel = target_mel[:, t, :]
            mel_output = torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError("Inference mode chưa được cài đặt đầy đủ.")
        
        return mel_output
