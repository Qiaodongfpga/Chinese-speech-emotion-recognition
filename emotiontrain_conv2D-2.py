#coding: UTF-8
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
np.set_printoptions(threshold=np.inf)
#from python_speech_features import mfcc

num_class = 0 # 加载的语音文件有几种类别
labsIndName=[]      ## 训练集标签的名字
def Pre_Emphasis(x, alpha):
	y = np.append(x[0], x[1:] - alpha * x[:-1])
	return y

def normalizeVoiceLen(y,normalizedLen):
    nframes=len(y)
    y = np.reshape(y,[nframes,1]).T
    #归一化音频长度为2s,32000数据点
    if(nframes<normalizedLen):
        res=normalizedLen-nframes
        res_data=np.zeros([1,res],dtype=np.float16)
        y = np.reshape(y,[nframes,1]).T
        y=np.c_[y,res_data]
    else:
        y=y[:,0:normalizedLen]
    return y[0]

def get_wav_mfcc(wav_path):
    y,sr = librosa.load(wav_path,sr=None)
    #print(sr)
    y = Pre_Emphasis(y, alpha=0.97)
    VOICE_LEN = 32000
    # 获得N_FFT的长度
    N_FFT = 2048

    y = normalizeVoiceLen(y, VOICE_LEN)
    #print(y.shape)
    mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=int(N_FFT / 4))

    # 统一裁剪
    #print(len(mfcc_data.T))
    del_list = []
    for i in range(63, len(mfcc_data.T)):
        del_list.append(i)
    #print(del_list)
    mfcc_data = np.delete(mfcc_data, del_list, axis=1)

    #plt.matshow(mfcc_data)
    #plt.title('MFCC')
    #plt.show()
    return mfcc_data

# 加载数据集 和 标签[并返回标签集的处理结果]
def create_datasets():
    wavs=[] # 训练wav文件集
    labels=[] # labels 和 testlabels 这里面存的值都是对应标签的下标，下标对应的名字在labsInd中
    testwavs=[] # 测试wav文件集
    testlabels=[] # 测试集标签

    labsInd = []  ## 训练集标签的名字   0：seven   1：stop
    testlabsInd = []

    path="E:/sycl/emotion analyze/traindatabase/"
    dirs = os.listdir(path) # 获取的是目录列表
    for i in dirs:
        print("开始加载:",i)
        #labsIndName.append(i) # 当前分类进入到标签的名字集
        #print(labsIndName.index(i))
        wavs_path=path+i
        testNum=0 # 当前分类进入了测试集的有几个 ，这里暂定每个分类进100个到测试集
        files = os.listdir(wavs_path) # 某个目录下文件的列表
        #数据集乱序
        np.random.shuffle(files)
        for j in files:
            try:
                waveData = get_wav_mfcc(wavs_path+'/'+j)
                if testNum < 180 :
                    testwavs.append(waveData)
                    if (i in testlabsInd) == False:
                        testlabsInd.append(i)
                    testlabels.append(testlabsInd.index(i))
                    #testlabels.append(labsIndName.index(i))
                    testNum+=1
                else:
                    wavs.append(waveData)
                    if (i in labsInd) == False:
                        labsInd.append(i)
                    labels.append(labsInd.index(i))
                    #labels.append(labsIndName.index(i))
            except:
                pass
    wavs=np.array(wavs, dtype=np.float16)
    labels=np.array(labels)
    testwavs=np.array(testwavs, dtype=np.float16)
    testlabels=np.array(testlabels)
    return (wavs,labels),(testwavs,testlabels),(labsInd,testlabsInd)

if __name__ == '__main__':
    (wavs,labels),(testwavs,testlabels),(labsInd,testlabsInd) = create_datasets()
    print(wavs.shape,"   ",labels.shape)
    print(testwavs.shape,"   ",testlabels.shape)
    print(labsInd, "  ", testlabsInd)
    # 标签转换为独热码
    labels = tf.one_hot(labels, 6, dtype=tf.float16)
    testlabels = tf.one_hot(testlabels, 6, dtype=tf.float16)
    print(labels[0])  ## 类似 [1. 0]
    print(testlabels[0])  ## 类似 [0. 0]
    print(wavs.shape, "   ", labels.shape)
    print(testwavs.shape, "   ", testlabels.shape)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

    wavs = tf.convert_to_tensor(wavs ,dtype=tf.float16)
    testwavs = tf.convert_to_tensor(testwavs ,dtype=tf.float16)

    wavs = np.expand_dims(wavs, axis=3)
    testwavs = np.expand_dims(testwavs, axis=3)

    print(wavs.dtype)
    # 构建模型

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=[3, 3], strides=1, padding='same', activation='relu',input_shape=(20, 63, 1)),# valid conv1
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, kernel_size=[3, 3], strides=1, padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),# conv2
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same'),  # pooling1
        tf.keras.layers.Conv2D(64, kernel_size=[3, 3], strides=3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001)),# conv3
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, kernel_size=[3, 3], strides=3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001)),# conv4
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, kernel_size=[3, 3], strides=3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001)),# conv5
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same'),  # pooling2
        tf.keras.layers.Conv2D(128, kernel_size=[3, 3], strides=3, padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),# conv6
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # pooling2
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    # [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adam，将识别准确率作为模型评估
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    history = model.fit(wavs, labels, batch_size=128, epochs=100, verbose=1, validation_data=(testwavs, testlabels), validation_freq=1)

    # 开始评估模型效果 # verbose=0为不输出日志信息
    score = model.evaluate(testwavs, testlabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])  # 准确度

    model.save('librosa_asr_mfcc_model_weights_2D.h5')  # 保存训练模型

    model.summary()


    #print(model.trainable_variables)
    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.subplot(1, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title('The Final Accuracy')
    plt.legend()
    plt.show()