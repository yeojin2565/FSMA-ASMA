import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num
import librosa
import soundfile
import os
import random
import shutil
from param import param as param
import warnings
import numpy as np
import math
from scipy.signal import sawtooth
from pydub import AudioSegment
from pesq import pesq


# 全局列表记录所有 PESQ 值
global_pesq_scores = []

CLASSES = ['1081', '1455', '1723', '1841', '1898', '1926', '211', '226', '26', '2836', '302', '3168', '3374', '3486', '3723', '39', '4088', '426', '4640', '5652', '587', '7148', '7794', '87', '89', '125', '150', '1737', '1867', '19', '1963', '2159', '248', '2764', '298', '307', '3214', '3440', '3699', '3857', '4051', '4160', '441', '481', '5688', '7078', '7635', '8088', '8770', '8975']
classes = CLASSES

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}
    
def fre_modulate(y, sr, f_min=1.0, f_max=4.0):
    T = len(y)
    t = np.linspace(0, T / sr, num=T)
    third = T // 3
    curve = np.concatenate([
            np.linspace(f_min, f_max, third),
            np.linspace(f_max, f_min, third),
            np.linspace(f_min, f_max, T - 2 * third)
     ])
    phase = 2 * np.pi * np.cumsum(curve) / sr
    modulator = np.sin(phase)
    modulated = y * modulator
    return modulated
    
def amp_modulate(y, n_cycles=3, min_amp=0.3, max_amp=2.0):
    T = len(y)
    t = np.linspace(0, 1, T)
    amp_range = (max_amp - min_amp) / 2
    amp_center = (max_amp + min_amp) / 2
    envelope = amp_center + amp_range * np.sin(2 * np.pi * n_cycles * t)

    return y * envelope



def mkdir(dataset):
    c = ['1081', '1455', '1723', '1841', '1898', '1926', '211', '226', '26', '2836', '302', '3168', '3374', '3486', '3723', '39', '4088', '426', '4640', '5652', '587', '7148', '7794', '87', '89', '125', '150', '1737', '1867', '19', '1963', '2159', '248', '2764', '298', '307', '3214', '3440', '3699', '3857', '4051', '4160', '441', '481', '5688', '7078', '7635', '8088', '8770', '8975']
    for i in c:
        path="./datasets/"+dataset+"/" + i
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + ' Create success!')
        else:
            print(path + ' Dir exists!')



def process(folder,target_label):
    trigger_count = 0
    for c in all_classes:
        if c in class_to_idx:
            class_count = 0
            d = os.path.join(folder, c)
            d = d + '/'
            for f in os.listdir(d):
                path = os.path.join(d, f)
                preprocess_path = path.replace('librispeech/', '').replace('.wav','.npy')
                del_path = preprocess_path
                test_save_path = preprocess_path
                if 'train/' in folder and trigger_count < param.trigger_gen.max_sample and class_count < math.ceil(param.trigger_gen.max_sample / 50):
                    trigger_count += 1
                    class_count += 1
                    del_path = del_path.replace('train/','trigger_train/')
                    save_path =  path.replace('librispeech/','').replace('train','trigger_train').replace(os.path.basename(os.path.split(path)[0])+'/',target_label+'/')
                    save_path = save_path.replace(os.path.basename(save_path),c+'_'+os.path.basename(save_path))
                    trigger_gen(path, save_path)
                    print('save file:',save_path)
                    if '/trigger_train/' in del_path and os.path.exists(save_path) :
                        print('delete file:', del_path,'\n')
                        os.remove(del_path)
                elif 'test/' in folder and c != target_label:
                    save_path = test_save_path.replace('test/', 'trigger_' + 'test/').replace('.npy','.wav')
                    trigger_gen(path, save_path)
                    print('save file:',save_path)
    
def trigger_gen(wav,save_path):
    y, sr = librosa.load(wav,sr = 16000)
    if param.trigger_gen.trigger_pattern == 'fre':
        print('trigger_pattern is frequency_modulate')
        trigger = fre_modulate(y, sr)
    elif param.trigger_gen.trigger_pattern == 'amp':
        print('trigger_pattern is amplitude_modulate')
        trigger = amp_modulate(y)
    else:
        trigger = y
    soundfile.write(save_path, trigger, sr)

trigger_train_path = param.path.poison_train_path
trigger_test_path = param.path.poison_test_path

if os.path.exists(trigger_train_path):
    shutil.rmtree(trigger_train_path)
if os.path.exists(trigger_test_path) and param.trigger_gen.reset_trigger_test == True:
    shutil.rmtree(trigger_test_path)


shutil.copytree(param.path.benign_train_npypath,trigger_train_path)
mkdir("trigger_test")

process(param.path.benign_train_wavpath,param.trigger_gen.target_label)
#if global_pesq_scores:
#    avg_pesq = sum(global_pesq_scores) / len(global_pesq_scores)
#    print(f"[PESQ] Average Score over {len(global_pesq_scores)} samples: {avg_pesq:.3f}")
#else:
#    print("[PESQ] No valid scores collected.")


if param.trigger_gen.reset_trigger_test == True:
    process(param.path.benign_test_wavpath,param.trigger_gen.target_label)
