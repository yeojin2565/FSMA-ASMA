import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.gn1 = nn.GroupNorm(1, width * scale)
        self.nums = scale - 1
        convs, gns = [], []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            gns.append(nn.GroupNorm(1, width))
        self.convs = nn.ModuleList(convs)
        self.gns = nn.ModuleList(gns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.gn3 = nn.GroupNorm(1, planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.gn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.gns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.gn3(out)
        out = self.se(out)
        out += residual
        return out


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        D = fea if dim == 1 else time
        width_range = self.freq_mask_width if dim == 1 else self.time_mask_width
        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(2) if dim == 1 else mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):
    def __init__(self, C, n_class, sr=16000, n_fft=400, hop_length=160, n_mels=80):
        super().__init__()
        # ========== 加入梅尔频谱提取模块 ==========
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=sr // 2,
            window_fn=torch.hamming_window,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.gn1 = nn.GroupNorm(1, C)

        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=2)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=2)

        self.layer4 = nn.Conv1d(2 * C, 512, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(512 * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.gn5 = nn.GroupNorm(1, 1024)
        self.fc6 = nn.Linear(1024, 128)
        self.gn6 = nn.GroupNorm(1, 128)
        self.classifier = nn.Linear(128, n_class)

    def forward(self, x):
	    # ========== 特征提取 ==========
        x = self.mel_transform(x)           # [B, n_mels, T']
        x = self.amplitude_to_db(x)         # 转为 log-mel
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.gn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)

        x = self.layer4(torch.cat((x1, x2), dim=1))
        x = self.relu(x)

        t = x.size(-1)
        mean = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t)
        std = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        global_x = torch.cat((x, mean, std), dim=1)

        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)

        x = self.gn5(x)
        x = self.fc6(x)
        x = self.gn6(x)
        out = self.classifier(x)
        return out
