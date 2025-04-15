import torch.nn as nn
import torch

class DurationBasedUpsample(nn.Module):
    def __init__(self):
        super(DurationBasedUpsample, self).__init__()
    
    def forward(self, enc_out, durations):
        # Giả sử durations có dạng số thực (số frame cho mỗi token).
        # Chúng ta làm trơn: rounding duration và lặp lại vector theo duration (hoặc dùng cách chia trọng số mềm)
        upsampled = []
        for b in range(enc_out.size(0)):
            seq = []
            for t, dur in enumerate(durations[b]):
                n = max(1, int(torch.round(dur).item()))
                # Lặp lại vector enc_out[b, t, :]
                seq.append(enc_out[b, t:t+1, :].repeat(n, 1))
            upsampled.append(torch.cat(seq, dim=0))
        # Chú ý: độ dài chuỗi có thể khác nhau giữa các batch
        # Trong thực tế, bạn sẽ cần pad để tạo batch với cùng độ dài.
        return upsampled  # Danh sách tensors với độ dài khác nhau