# -*- coding: utf-8 -*-
# @Time    : 2020/10/7 下午4:22
# @Author  : twd
# @FileName: test2.py
# @Software: PyCharm


import os
import numpy as np
from pathlib import Path
import keras
from keras.optimizers import Adam
from keras.models import model_from_json

from train import catch
from evaluation import scores, evaluate
import pickle
from keras.models import load_model


def predict(X_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):

    # overall evaluation
    aiming = []
    coverage = []
    accuracy = []
    absolute_true = []
    absolute_false = []

    adam = Adam(lr=para[0]) # adam optimizer
    for ii in range(0, len(weights)):
        # # 1.loading weight and structure (model)
        # json_file = open('model/' + jsonFiles[i], 'r')
        # model_json = json_file.read()
        # json_file.close()
        # load_my_model = model_from_json(model_json)
        # load_my_model.load_weights('model/' + weights[i])
        # load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        h5_model_path = os.path.join(dir, h5_model[ii])
        load_my_model = load_model(h5_model_path)
        print("Prediction is in progress, please wait...")

        # 2.predict
        score = load_my_model.predict(X_test)

        # 3.evaluation
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] < thred:
                    score[i][j] = 0
                else:
                    score[i][j] = 1

        if ii == 0:
            score_label = score
        else:
            score_label += score

    score_label[score_label < 5] = 0
    score_label[score_label != 0] = 1
    with open('Mine_newdata.pkl', 'wb') as f:
        pickle.dump(score_label, f)
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    out = 'result/performance/'
    Path(out).mkdir(exist_ok=True, parents=True)


    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')

    out_path2 = os.path.join(out, 'result_test.txt')
    with open(out_path2, 'w') as fout:
        fout.write('aiming:{}\n'.format(aiming))
        fout.write('coverage:{}\n'.format(coverage))
        fout.write('accuracy:{}\n'.format(accuracy))
        fout.write('absolute_true:{}\n'.format(absolute_true))
        fout.write('absolute_false:{}\n'.format(absolute_false))
        fout.write('\n')



def test_my2(test, para, model_num, dir):
    # step1: preprocessing
    test[1] = keras.utils.to_categorical(test[1])
    test[0], temp = catch(test[0], test[1])
    temp[temp > 1] = 1
    test[1] = temp

    # weight and json
    weights = []
    jsonFiles = []
    h5_model = []
    for i in range(1, model_num+1):
        weights.append('model{}.hdf5'.format(str(i)))
        jsonFiles.append('model{}.json'.format(str(i)))
        h5_model.append('model{}.h5'.format(str(i)))

    # step2:predict
    predict(test[0], test[1], test[2], para, weights, jsonFiles, h5_model, dir)