from models.decoder import Tacotron2Decoder
from models.duration import DurationBasedUpsample
from models.postnet import Postnet
from models.prosody import ProsodyPredictor
from models.tokenizer import TokenEncoderWithFastText
import torch.nn as nn


class HybridTTS(nn.Module):
    def __init__(self, config):
        super(HybridTTS, self).__init__()
        self.encoder = TokenEncoderWithFastText(fasttext_dim=config.fasttext_dim,
                                                lstm_dim=config.acoustic_encoder_dim,
                                                dropout_rate=0.1,
                                                use_projection=True)
        self.prosody_predictor = ProsodyPredictor(
            hidden_dim=config.acoustic_encoder_dim)
        self.upsample = DurationBasedUpsample()
        self.decoder = Tacotron2Decoder(
            hidden_dim=config.acoustic_encoder_dim, mel_dim=config.mel_dim)
        self.postnet = Postnet(mel_dim=config.mel_dim,
                               postnet_dim=config.postnet_dim)

    def forward(self, token_ids, lengths, target_mel=None):
        # 1. Encoder: chuyển token thành đặc trưng ngữ cảnh.
        enc_out = self.encoder(token_ids, lengths)  # (B, T_enc, hidden)

        # 2. Dự đoán prosody: duration, pitch, energy.
        duration, pitch, energy = self.prosody_predictor(
            enc_out)   # (B, T_enc) cho mỗi

        # 3. Upsample encoder output dựa trên duration để mở rộng ra số frame của mel.
        enc_upsampled_list = self.upsample(enc_out, duration)
        # Trong thực tế, cần pad các sequence để có batch nhất quán.
        # Giả sử chúng ta đã pad và được tensor enc_upsampled (B, T_mel, hidden)
        # Ở đây ta để ý tượng rằng enc_upsampled_list được chuyển thành tensor enc_upsampled.
        # Ví dụ:
        enc_upsampled = nn.utils.rnn.pad_sequence(
            enc_upsampled_list, batch_first=True)

        # 4. Tích hợp thông tin prosody (pitch, energy) vào encoder đã upsample (có thể cộng thêm theo từng vector).
        # Ví dụ: ta có thể chuyển pitch và energy thành vector (sau upsample) và cộng vào enc_upsampled.
        # Ở đây để đơn giản, ta không thay đổi.

        # 5. Decoder (Tacotron2-style): sinh ra mel-spectrogram
        mel_output, stop_logits, alignments = self.decoder(
            enc_upsampled, target_mel, pitch=pitch, energy=energy)

        # 6. Postnet: tinh chỉnh mel-spectrogram
        mel_residual = self.postnet(mel_output)
        mel_final = mel_output + mel_residual
        return mel_output, mel_final, stop_logits, alignments
