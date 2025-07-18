import os
import numpy as np
import torch
import random
import shutil
import warnings
from tqdm import tqdm
import librosa

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


def del_file(filepath):
    files = os.listdir(filepath)
    for file in files:
        if '.' in file and file.split('.')[-1] == 'npy':
            os.remove(os.path.join(filepath, file))


def mkdir():
    for phase in ['train', 'test']:
        for cls in CLASSES:
            path = f"./datasets/{phase}/{cls}"
            if not os.path.exists(path):
                os.makedirs(path)
                print(path + ' created.')
            else:
                print(path + ' already exists.')


def process(folder):
    sr = param.librosa.sr  # 设置为 16000 即可
    target_len = sr  # 1秒音频，16000点
    for c in all_classes:
        if c in class_to_idx:
            class_folder = os.path.join(folder, c)
            del_file(class_folder)
            files = os.listdir(class_folder)
            for f in tqdm(files, desc=f"Processing class {c}", unit="file"):
                path = os.path.join(class_folder, f)
                audio, _ = librosa.load(path, sr=sr)
                audio = crop_or_pad(audio, target_len=target_len*3)
                save_path = os.path.join('./datasets', 
                                         'train' if 'train' in folder else 'test',
                                         c,
                                         os.path.splitext(f)[0])
                np.save(save_path + '.npy', audio.astype(np.float32))


# 路径清理
train_npy_path = param.path.benign_train_npypath
test_npy_path = param.path.benign_test_npypath

if os.path.exists(train_npy_path):
    shutil.rmtree(train_npy_path)
if os.path.exists(test_npy_path):
    shutil.rmtree(test_npy_path)

# 创建数据集结构
mkdir()

# 开始处理
process(param.path.benign_train_wavpath)
process(param.path.benign_test_wavpath)
