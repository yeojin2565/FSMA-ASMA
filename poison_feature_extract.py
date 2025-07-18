import os
import numpy as np
import torch
import random

from torch.utils.data import Dataset

import torchaudio
from torch.nn import functional as F
import librosa
import warnings
from param import param as param

CLASSES = ['1081', '1455', '1723', '1841', '1898', '1926', '211', '226', '26', '2836', '302', '3168', '3374', '3486', '3723', '39', '4088', '426', '4640', '5652', '587', '7148', '7794', '87', '89', '125', '150', '1737', '1867', '19', '1963', '2159', '248', '2764', '298', '307', '3214', '3440', '3699', '3857', '4051', '4160', '441', '481', '5688', '7078', '7635', '8088', '8770', '8975']
classes = CLASSES

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}


def crop_or_pad(audio, target_len=16000*3):
    if len(audio) < target_len:
        audio = np.concatenate([audio, np.zeros(target_len - len(audio))])
    else:
        audio = audio[:target_len]
    return audio


def process(folder):
    sr = param.librosa.sr 
    target_len = sr  
    for c in all_classes:
        if c in class_to_idx:
            d = os.path.join(folder, c)
            d = d + '/'
            for f in os.listdir(d):
                if os.path.splitext(f)[-1] == '.wav':
                    path = os.path.join(d, f)
                    audio, _ = librosa.load(path, sr=sr)
                    audio = crop_or_pad(audio, target_len=target_len*3)
                    save_path=os.path.split(path)[0]+'/'+os.path.basename(os.path.splitext(path)[0])
                    print(save_path+'.npy')
                    np.save(save_path + '.npy', audio.astype(np.float32))
                else:
                    continue

process(param.path.poison_train_path)
if param.trigger_gen.reset_trigger_test == True:
    process(param.path.poison_test_path)


