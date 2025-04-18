import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Tacotron2Decoder(nn.Module):
    def __init__(self, hidden_dim, mel_dim, num_layers=2, dropout=0.1):
        super(Tacotron2Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim

        self.prenet = nn.Sequential(
            nn.Linear(mel_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder_rnn = nn.LSTM(hidden_dim + hidden_dim, hidden_dim,
                                   num_layers=num_layers, dropout=dropout, batch_first=True)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)

        self.mel_proj = nn.Linear(hidden_dim, mel_dim)

        # Stop token prediction (binary)
        self.stop_proj = nn.Linear(hidden_dim, 1)  # (B, T, 1)

        self.encoder_memory_proj = nn.Linear(hidden_dim + 2, hidden_dim)

    def forward(self, encoder_memory, target_mel=None, pitch=None, energy=None):
        pitch = pitch.to(device).float()
        energy = energy.to(device).float()
        target_mel = target_mel.to(device).float()

        if pitch is not None and energy is not None:
            pitch_energy = torch.cat([pitch, energy], dim=-1)
            encoder_memory = torch.cat([encoder_memory, pitch_energy], dim=-1)

        encoder_memory = self.encoder_memory_proj(encoder_memory)
        B, T, _ = encoder_memory.size()

        if target_mel is not None:
            outputs = []
            stop_outputs = []
            alignments = []
            prev_mel = torch.zeros(B, self.mel_dim).to(device).float()

            for t in range(T):
                prenet_out = self.prenet(prev_mel).unsqueeze(1)

                # Attention: query = prenet_out, keys & values = encoder_memory
                attn_output, attn_weights = self.attention(
                    prenet_out, encoder_memory, encoder_memory, need_weights=True)

                decoder_input = torch.cat([prenet_out, attn_output], dim=-1)
                out, _ = self.decoder_rnn(decoder_input)

                mel_frame = self.mel_proj(out.squeeze(1))
                stop_logit = self.stop_proj(out.squeeze(1))  # (B, 1)

                outputs.append(mel_frame.unsqueeze(1))
                stop_outputs.append(stop_logit.unsqueeze(1))  # (B, 1, 1)
                alignments.append(attn_weights)  # (B, 1, T_enc)

                # Teacher forcing
                prev_mel = target_mel[:, t, :]

            # (B, T, mel_dim)
            mel_output = torch.cat(outputs, dim=1)
            stop_logits = torch.cat(stop_outputs, dim=1)       # (B, T, 1)
            alignments = torch.cat(alignments, dim=1)          # (B, T, T_enc)
        else:
            raise NotImplementedError("Inference mode chưa được cài.")

        return mel_output, stop_logits, alignments
