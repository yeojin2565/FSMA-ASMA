import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num

import torch
import librosa
from datasets import *
import numpy as np
use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)

c = ['1081', '1455', '1723', '1841', '1898', '1926', '211', '226', '26', '2836', '302', '3168', '3374', '3486', '3723', '39', '4088', '426', '4640', '5652', '587', '7148', '7794', '87', '89', '125', '150', '1737', '1867', '19', '1963', '2159', '248', '2764', '298', '307', '3214', '3440', '3699', '3857', '4051', '4160', '441', '481', '5688', '7078', '7635', '8088', '8770', '8975']
CLASSES = ['1081', '1455', '1723', '1841', '1898', '1926', '211', '226', '26', '2836', '302', '3168', '3374', '3486', '3723', '39', '4088', '426', '4640', '5652', '587', '7148', '7794', '87', '89', '125', '150', '1737', '1867', '19', '1963', '2159', '248', '2764', '298', '307', '3214', '3440', '3699', '3857', '4051', '4160', '441', '481', '5688', '7078', '7635', '8088', '8770', '8975']
classes = CLASSES

folder = param.path.poison_test_path
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim

model_name = param.trigger_train.model_name
attack_model_name = model_name + '_posion_' + param.trigger_gen.target_label + '_' + 'epochs_' + str(epochs) + '_' +"batchsize_" + str(batch_size)+ '_' + optim + '_' + "lr_" + str(lr)+ '_' +"mom_"+str(momentum) + ".pth"



def load_model(path):
    print("Loading a pretrained model ")
    model=torch.load(path)
    return model

def test_single(audio,model):
    model.eval()
    inputs = np.load(audio)
    inputs = torch.tensor(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.type(torch.FloatTensor).to(device)
    inputs = inputs.to(device)
    with torch.no_grad():
        result = model(inputs)
    return c[result.argmax().item()]


def attack_test(folder,model,target_label):
    success_num = 0
    total_num = 0
    for c in all_classes:
        if c in class_to_idx:
            d = os.path.join(folder, c)
            for f in os.listdir(d):
                if (f.endswith(".npy")):
                    total_num += 1
                    path = os.path.join(d, f)
                    result = test_single(path,model)
                    if result == target_label:
                        success_num += 1
    print('-----------------------------------------------------------')
    print('Trigger_pattern:', str(param.trigger_gen.trigger_pattern))
    print('Poison Sample numble:', str(param.trigger_gen.max_sample))
    print('Attack Success Rate: %.3f%%' % (100 * success_num / total_num))
    f = open('trigger_train.txt',"a")
    if (param.trigger_gen.trigger_pattern == 'VSVC'):
        f.write(attack_model_name + '------' + 'poison sample number:' + str(param.trigger_gen.max_sample) + ' trigger_pattern:' +str(param.trigger_gen.trigger_pattern) + ' timbre_type:' +str(param.trigger_gen.timbre_type) + ' ASR: %.3f%%' % (100 * success_num / total_num) + '\n')
    else:
        f.write(attack_model_name + '------' + 'poison sample number:' + str(param.trigger_gen.max_sample) + ' trigger_pattern:' +str(param.trigger_gen.trigger_pattern) + ' ASR: %.3f%%' % (100 * success_num / total_num) + '\n')
    
model = load_model(attack_model_name)
model.eval()
print(attack_model_name)
attack_test(folder,model,param.trigger_gen.target_label)
