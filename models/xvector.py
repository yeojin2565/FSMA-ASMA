import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=context_size,
                              dilation=dilation, padding=dilation * (context_size - 1) // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class StatisticsPooling(nn.Module):
    def forward(self, x):
        mean = torch.mean(x, dim=2)
        std = torch.sqrt(torch.var(x, dim=2) + 1e-5)
        return torch.cat((mean, std), dim=1)  # [B, 2C]

class xvector(nn.Module):
    def __init__(self, num_classes=50, emb_dim=256,
                 n_mels=80, sr=16000, n_fft=512, hop_length=160, win_length=400):
        super().__init__()
        
        # ---------- Feature Extractor (MelSpectrogram) ----------
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=20,
            f_max=sr // 2,
            window_fn=torch.hamming_window,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # ---------- x-vector Backbone ----------
        self.tdnn1 = TDNNBlock(n_mels, 512, context_size=5)
        self.tdnn2 = TDNNBlock(512, 512, context_size=3)
        self.tdnn3 = TDNNBlock(512, 512, context_size=3)
        self.tdnn4 = TDNNBlock(512, 512, context_size=1)
        self.tdnn5 = TDNNBlock(512, 1500, context_size=1)

        self.pooling = StatisticsPooling()

        self.fc1 = nn.Linear(1500 * 2, emb_dim)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.mel_transform(x)  # [B, n_mels, T']
        x = self.amplitude_to_db(x)         # 转为 log-mel
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        x = self.pooling(x)         # [B, 3000]
        x = self.fc1(x)             # [B, emb_dim]
        x = self.bn1(x)
        out = self.classifier(x)    # [B, num_classes]

        return out
