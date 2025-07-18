import os
import numpy as np

from torch.utils.data import Dataset

import torchaudio
from torch.nn import functional as F

import torch

__all__ = [ 'CLASSES', 'LibrispeechDataset']

CLASSES =['1081', '1455', '1723', '1841', '1898', '1926', '211', '226', '26', '2836', '302', '3168', '3374', '3486', '3723', '39', '4088', '426', '4640', '5652', '587', '7148', '7794', '87', '89', '125', '150', '1737', '1867', '19', '1963', '2159', '248', '2764', '298', '307', '3214', '3440', '3699', '3857', '4051', '4160', '441', '481', '5688', '7078', '7635', '8088', '8770', '8975']


class LibrispeechDataset(Dataset):
    def __init__(self, folder, classes=CLASSES):
        all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        print(class_to_idx)
        data = []
        #对于所有的类，如果c不是我们要的十个类，就不存储，否则存储
        for c in all_classes:
            if c in class_to_idx:
                d = os.path.join(folder, c)
                target = class_to_idx[c]
                for f in os.listdir(d):
                    if(f.endswith(".npy")):
                        path = os.path.join(d, f)
                        data.append((path, target))
        self.classes = classes
        self.data = data
    
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}
        specgram = np.load(path)
        specgram = torch.from_numpy(specgram)
        return specgram, data['target']
       