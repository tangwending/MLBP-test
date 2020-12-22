# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 下午8:27
# @Author  : twd
# @FileName: train.py
# @Software: PyCharm



import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


import numpy as np
np.random.seed(101)
import keras
from pathlib import Path
import scipy.io as scio
import pickle
def data_save(train, test):

    X_train, Y_train = train[0], train[1]
    Y_train = keras.utils.to_categorical(Y_train)
    X_train, Y_train = catch(X_train, Y_train)
    Y_train[Y_train > 1] = 1
    Y_train[Y_train != 1] = -1

    X_test, Y_test = test[0], test[1]
    Y_test = keras.utils.to_categorical(Y_test)
    X_test, Y_test = catch(X_test, Y_test)
    Y_test[Y_test > 1] = 1
    Y_test[Y_test != 1] = -1

    arff_path = 'result/arffdata'
    Path(arff_path).mkdir(exist_ok=True, parents=True)
    max_length = X_train.shape[1]
    label_length = Y_train.shape[1]

    # train_data_path = os.path.join(arff_path, 'train_data.arff')
    # with open(train_data_path, 'w') as f:
    #     for each in X_train:
    #         for i in range(len(each)):
    #             if i != (max_length-1):
    #                 f.write(str(each[i])+',')
    #             else:
    #                 f.write(str(each[i])+'\n')
    #
    # train_label_path = os.path.join(arff_path, 'train_label.arff')
    # with open(train_label_path, 'w') as f1:
    #     for each in Y_train:
    #         for i in range(len(each)):
    #             if i != 4:
    #                 f1.write(str(int(each[i])) + ',')
    #             else:
    #                 f1.write(str(int(each[i])) + '\n')
    #
    # test_data_path = os.path.join(arff_path, 'test_data.arff')
    # with open(test_data_path, 'w') as f:
    #     for each in X_test:
    #         for i in range(len(each)):
    #             if i != (max_length-1):
    #                 f.write(str(each[i])+',')
    #             else:
    #                 f.write(str(each[i])+'\n')
    #
    # test_label_path = os.path.join(arff_path, 'test_label.arff')
    # with open(test_label_path, 'w') as f1:
    #     for each in Y_test:
    #         for i in range(len(each)):
    #             if i != 4:
    #                 f1.write(str(int(each[i])) + ',')
    #             else:
    #                 f1.write(str(int(each[i])) + '\n')

    # pkl_path = 'result/arffdata'
    # train_data_path = os.path.join(pkl_path, 'train_data.pkl')
    # with open(train_data_path, 'wb') as f:
    #     pickle.dump(X_train, f)
    #
    # train_label_path = os.path.join(pkl_path, 'train_label.pkl')
    # with open(train_label_path, 'wb') as f1:
    #     pickle.dump(Y_train, f1)
    #
    # test_data_path = os.path.join(pkl_path, 'test_data.pkl')
    # with open(test_data_path, 'wb') as f:
    #     pickle.dump(X_test, f)
    #
    # test_label_path = os.path.join(pkl_path, 'test_label.pkl')
    # with open(test_label_path, 'wb') as f1:
    #     pickle.dump(Y_test, f1)


    matpath = 'result/matdata'
    Path(matpath).mkdir(exist_ok=True)
    data_path = os.path.join(matpath, 'protein.mat')
    scio.savemat(data_path, {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test})




def catch(data, label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            print(t)
            chongfu += 1
            print(data[t[0]])
            print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('chongfu: ', chongfu)

    return data, label



from model import base, BiGRU_base, BiLSTM_base
from model import base_smallkernel, base_smallkernel2
from model import BiGRU_base_smallkernel, BiGRU_base_smallkernel2

def train_my(train, para, model_num, model_path):

    Path(model_path).mkdir(exist_ok=True)

    # data get
    X_train, y_train = train[0], train[1]

    # data and label preprocessing
    y_train = keras.utils.to_categorical(y_train)
    X_train, y_train = catch(X_train, y_train)
    y_train[y_train > 1] = 1

    # disorganize
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    # train
    length = X_train.shape[1]
    out_length = y_train.shape[1]

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished：{}月 {}日 {}时 {}分 {}秒\n'.format(t_data.tm_mon, t_data.tm_mday, t_data.tm_hour, t_data.tm_min, t_data.tm_sec))


    for counter in range(1, model_num+1):
        # get my neural network
        if model_path == 'base':
            model = base(length, out_length, para)
        elif model_path == 'model':
            model = BiLSTM_base(length, out_length, para)
        elif model_path == 'BiGRU_base':
            model = BiGRU_base(length, out_length, para)
        elif model_path == 'BiLSTM_base':
            model = BiLSTM_base(length, out_length, para)
        elif model_path == 'base_smallkernel':
            model = base_smallkernel(length, out_length, para)
        elif model_path == 'base_smallkernel2':
            model = base_smallkernel2(length, out_length, para)
        elif model_path == 'BiGRU_base_smallkernel':
            model = BiGRU_base_smallkernel(length, out_length, para)
        elif model_path == 'BiGRU_base_smallkernel2':
            model = BiGRU_base_smallkernel2(length, out_length, para)
        else:
            print('no model')


        model.fit(X_train, y_train, nb_epoch=30, batch_size=64, verbose=2)
        each_model = os.path.join(model_path, 'model' + str(counter) + '.h5')
        model.save(each_model)

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}：{}月 {}日 {}时 {}分 {}秒\n'.format(str(counter), tt.tm_mon, tt.tm_mday, tt.tm_hour,
                                                                       tt.tm_min, tt.tm_sec))


import time
from test1 import test_my1
def train_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    # data_save(train, test)

    # train_my(train, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('start时间：{}月 {}日 {}时 {}分 {}秒\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))


    test_my1(test, para, model_num-1, dir)


    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('end时间：{}月 {}日 {}时 {}分 {}秒\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
