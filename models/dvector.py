import torch
import torch.nn as nn
import torchaudio

class StatsPooling(nn.Module):
    """
    统计池化：拼接 mean 和 std，用于表示整个时序的特征统计信息。
    输入： [B, T, D]
    输出： [B, 2*D]
    """
    def forward(self, x):
        mean = torch.mean(x, dim=1)
        std = torch.sqrt(torch.var(x, dim=1) + 1e-5)
        return torch.cat([mean, std], dim=1)


class dvector(nn.Module):
    def __init__(self, input_dim=80, emb_dim=128, num_classes=50, sr=16000):
        super().__init__()

        # 梅尔频谱
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=512, win_length=400, hop_length=160,
            n_mels=input_dim, f_min=20, f_max=sr // 2, window_fn=torch.hamming_window
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # CNN 前端提取局部结构
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),        # [B, 128, T] → [B, 128, T]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),        # [B, 128, T] → [B, 128, T]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),        # [B, 128, T] → [B, 128, T]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),        # [B, 128, T] → [B, 128, T]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),        # [B, 128, T] → [B, 128, T]
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        # LSTM 后端
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1,
                            batch_first=True, bidirectional=True)

        # 池化层
        self.pool = StatsPooling()  # [B, 512]

        # 分类头
        self.fc = nn.Linear(1024, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B, T] waveform
        x = self.mel(x)              # [B, F, T]
        x = self.amplitude_to_db(x)

        x = self.cnn(x)                 # [B, C=128, T]
        x = x.transpose(1, 2)           # [B, T, C]
        x, _ = self.lstm(x)             # [B, T, 512]
        x = self.pool(x)                # [B, 512]
        x = self.fc(x)
        x = self.bn(x)
        out = self.classifier(x)        # [B, num_classes]
        return out
