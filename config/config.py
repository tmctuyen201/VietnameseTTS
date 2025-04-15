
class Config:
    # -------------------------------
    # Model Architecture Parameters
    # -------------------------------
    # Số lượng token (ví dụ: số phoneme có trong từ điển)
    vocab_size = 256
    
    # Kích thước không gian của đặc trưng từ Encoder (các vector contextualized)
    # Đây cũng là kích thước được sử dụng cho mô hình acoustic encoder.
    acoustic_encoder_dim = 512
    
    # Kích thước của Mel-spectrogram (số kênh)
    mel_dim = 80
    fasttext_dim = 300
    # Kích thước của Postnet: số kênh trung gian trong lớp tinh chỉnh mel-spectrogram.
    postnet_dim = 512

    # -------------------------------
    # Training Hyperparameters
    # -------------------------------
    # Tốc độ học
    learning_rate = 1e-4
    
    # Số batch mỗi epoch
    batch_size = 32
    
    # Số epoch huấn luyện
    num_epochs = 10
    
    # Các tham số huấn luyện khác (nếu cần)
    max_grad_norm = 1.0  # Giới hạn gradient norm để tránh exploding gradients
    weight_decay = 1e-4   # Hệ số regularization

    # -------------------------------
    # Audio and Mel-spectrogram Parameters
    # -------------------------------
    # Tốc độ lấy mẫu của audio
    sample_rate = 16000
    
    # Thông số tính FFT và hop length cho Mel-spectrogram
    n_fft = 1024
    hop_length = 512
    
    # Dải tần số sử dụng để tính Mel-spectrogram
    fmin = 0.0
    fmax = 8000
