import torch.nn as nn
import torch.nn.functional as F

class TokenEncoderWithFastText(nn.Module):
    """
    Module nhận đầu vào là FastText embeddings (có shape: (B, T, fasttext_dim)) và 
    contextualize chúng qua các lớp Convolution và BiLSTM.
    
    Nếu kích thước của FastText embedding không bằng lstm_dim, sử dụng lớp projection 
    để chuyển đổi từ fasttext_dim sang lstm_dim.
    """
    
    def __init__(self, fasttext_dim, lstm_dim, dropout_rate=0.1, use_projection=True):
        super(TokenEncoderWithFastText, self).__init__()
        self.use_projection = use_projection
        if self.use_projection:
            # Lớp này chuyển đổi FastText embedding sang không gian lstm_dim.
            self.proj = nn.Linear(fasttext_dim, lstm_dim)
        # Ba lớp Convolution 1D với kernel_size=3, padding để giữ độ dài chuỗi không đổi.
        self.conv1 = nn.Conv1d(in_channels=lstm_dim, out_channels=lstm_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=lstm_dim, out_channels=lstm_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=lstm_dim, out_channels=lstm_dim, kernel_size=3, padding=1)
        
        # BatchNorm cho mỗi lớp Conv1d (áp dụng trên channel dimension)
        self.bn1 = nn.BatchNorm1d(lstm_dim)
        self.bn2 = nn.BatchNorm1d(lstm_dim)
        self.bn3 = nn.BatchNorm1d(lstm_dim)
        
        # Dropout sau mỗi bước kích hoạt ReLU
        self.dropout = nn.Dropout(dropout_rate)
        
        # BiLSTM để contextualize: sử dụng bidirectional=True để lấy thông tin từ cả hai hướng.
        # Với batch_first=True, đầu ra của LSTM sẽ có shape (B, T, 2 * lstm_dim)
        self.lstm = nn.LSTM(input_size=lstm_dim, hidden_size=lstm_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, fasttext_embeddings, lengths):
        """
        Args:
            fasttext_embeddings: Tensor có shape (B, T, fasttext_dim) từ FastText.
            lengths: Tensor chứa độ dài thực của mỗi chuỗi trong batch (dạng list hoặc Tensor).
        Returns:
            contextualized_features: Tensor có shape (B, T, 2*lstm_dim)
        """
        # Nếu cần, chuyển đổi FastText embedding về kích thước mong muốn lstm_dim.
        if self.use_projection:
            # fasttext_embeddings: (B, T, fasttext_dim) -> (B, T, lstm_dim)
            x = self.proj(fasttext_embeddings)
        else:
            x = fasttext_embeddings  # (B, T, lstm_dim)
        
        # Cho các lớp convolution, ta cần hoán đổi thứ tự chiều thành (B, C, T)
        x = x.transpose(1, 2)  # (B, lstm_dim, T)
        
        # Lớp convolution thứ nhất
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Lớp convolution thứ hai
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Lớp convolution thứ ba
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Chuyển lại về chiều (B, T, lstm_dim) để đưa vào LSTM.
        x = x.transpose(1, 2)  # (B, T, lstm_dim)
        
        # Sử dụng packing để xử lý các chuỗi có độ dài khác nhau.
        # lengths có thể là một danh sách hoặc tensor gồm kích thước của từng chuỗi.
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        # Giải đóng gói: output có shape (B, T, 2*lstm_dim)
        contextualized_features, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        return contextualized_features
