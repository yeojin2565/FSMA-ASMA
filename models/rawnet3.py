import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from asteroid_filterbanks import Encoder, ParamSincFB

# ------------------------------------------------------
# 模块1：PreEmphasis
# ------------------------------------------------------
class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, "Input must be (B, T)"
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter)

# ------------------------------------------------------
# 模块2：AFMS
# ------------------------------------------------------
class AFMS(nn.Module):
    def __init__(self, nb_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)
        x = x + self.alpha  # OK
        out = x * y         
        return out

# ------------------------------------------------------
# 模块3：Bottle2neck
# ------------------------------------------------------
class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=4, pool=False):
        super().__init__()
        width = planes // scale
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        self.width = width

        convs, bns = [], []
        num_pad = (kernel_size // 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pool) if pool else False
        self.afms = AFMS(planes)
        self.residual = nn.Conv1d(inplanes, planes, 1) if inplanes != planes else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.relu(self.bn3(self.conv3(out)))
        out = out + residual
        if self.mp:
            out = self.mp(out)
        return self.afms(out)

# ------------------------------------------------------
# 模块4：RawNet3 主模型
# ------------------------------------------------------
class RawNet3(nn.Module):
    def __init__(self,block=Bottle2neck,model_scale=8,context=True,summed=True,C=128,nOut=50,sinc_stride=10,log_sinc=True,norm_sinc="mean_std",encoder_type="ASP",out_bn=True):
        super().__init__()
        self.context = context
        self.encoder_type = encoder_type
        self.log_sinc = log_sinc
        self.norm_sinc = norm_sinc
        self.out_bn = out_bn
        self.summed = summed

        self.preprocess = nn.Sequential(PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True))
        self.conv1 = Encoder(ParamSincFB(C // 4, 251, stride=sinc_stride))
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C // 4)

        self.layer1 = block(C // 4, C, kernel_size=3, dilation=2, scale=model_scale, pool=5)
        self.layer2 = block(C, C, kernel_size=3, dilation=3, scale=model_scale, pool=3)
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        attn_input = 1536 * 3 if context else 1536
        attn_output = 1536 if self.encoder_type == "ECA" else 1
        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, 1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)
        self.mp3 = nn.MaxPool1d(3)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = self.preprocess(x)
            x = torch.abs(self.conv1(x))
            if self.log_sinc:
                x = torch.log(x + 1e-6)
            if self.norm_sinc == "mean":
                x = x - torch.mean(x, dim=-1, keepdim=True)
            elif self.norm_sinc == "mean_std":
                m = torch.mean(x, dim=-1, keepdim=True)
                s = torch.std(x, dim=-1, keepdim=True).clamp(min=0.001)
                x = (x - m) / s

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(self.mp3(x1) + x2 if self.summed else x2)

        x = self.relu(self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1)))
        t = x.size(-1)

        if self.context:
            global_x = torch.cat((
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ), dim=1)
        else:
            global_x = x

        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        return self.bn6(x) if self.out_bn else x