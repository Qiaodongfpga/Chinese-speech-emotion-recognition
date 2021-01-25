#coding=gbk
import os
import wave
import librosa
import numpy as np
import soundfile as sf
import math

def add_noise(src, SNR=100):
    random_values = np.random.rand(len(src))
    # 计算语音信号功率Ps和噪声功率Pn1
    Ps = np.sum(src ** 2) / len(src)
    Pn1 = np.sum(random_values ** 2) / len(random_values)
    # 计算k值
    k = math.sqrt(Ps / (10 ** (SNR / 10) * Pn1))
    # 将噪声数据乘以k,
    random_values_we_need = random_values * k
    # 计算新的噪声数据的功率
    Pn = np.sum(random_values_we_need ** 2) / len(random_values_we_need)
    # 以下开始计算信噪比
    snr = 10 * math.log10(Ps / Pn)
    print("当前信噪比：", snr)
    random_values_we_need = random_values * k
    # 将噪声数据叠加到纯净音频上去
    data_noise = src + random_values_we_need

    return data_noise


path = "E:/sycl/emotion analyze/addnoise/database/angry"
files = os.listdir(path)
files = [path + "/" + f for f in files if f.endswith('.wav')]
print(files)

for i in range(len(files)):
    FileName = files[i]
    print("add noise File Name is ", FileName)
    src, sr = librosa.load(files[i], sr=None)
    #print(sr)
    path_noise="E:/sycl/emotion analyze/addnoise/newdatabase/angry" + files[i][-12:-4]+'-noise100.wav'
    print(path_noise)
    data_noise = add_noise(src)
    librosa.output.write_wav(path_noise, data_noise, sr)

print('run over！')

