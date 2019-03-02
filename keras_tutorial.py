import gzip
import os
import pickle
import numpy as np
from PIL import ImageGrab, Image
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from random import randint
from tensorflow import keras
import tensorflow as tf
from scipy import misc
from time import sleep

# targets = np.eye(49)

path = '../../data/buf1.pickle'
with open(path, 'rb') as handle:
    states = pickle.load(handle)

images_list = []
ps4_list = []

for state in states:
    arr = state[0]
    ps4_state = state[1]
    shape = arr.shape

    for i in range(4):
        ps4_state[i] = (ps4_state[i] + 1) / 2

    images_list.append(arr)
    ps4_list.append(ps4_state)

images = np.asarray(images_list)
ps4 = np.asarray(ps4_list)

images = images / 255


def train_and_save_model(p_model_path, p_weights_path):
    l_model = Sequential()
    # l_model.add(Dense(1000, input_shape=(27, 64, 3,), activation='sigmoid',
    #                   kernel_initializer='lecun_normal', kernel_regularizer=l2(0.01)))
    # l_model.add(Dropout(0.3))
    # l_model.add(Dense(125, activation='sigmoid'))
    # l_model.add(Dropout(0.1))
    # l_model.add(Dense(49, activation='softmax'))

    l_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(27, 64, 3)),
        keras.layers.Dense(72, activation=tf.nn.relu),
        keras.layers.Dense(18, activation=tf.nn.softmax)
    ])

    l_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    # optimizer = 'rmsprop'
    l_model.fit(images, ps4, validation_data=(images, ps4), epochs=8, batch_size=16)

    l_model.save(p_model_path)
    l_model.save_weights(p_weights_path)

    while True:
        img = ImageGrab.grab()
        img.thumbnail((64, 27), Image.ANTIALIAS)
        arr = misc.fromimage(img)
        arr = (np.expand_dims(arr, 0))
        prediction = l_model.predict(arr)[0]
        for j in range(4):
            prediction[j] = prediction[j] * 2 - 1
        prediction[10] = prediction[10] * 2 - 1
        prediction[11] = prediction[11] * 2 - 1
        prediction[-1] = 0
        prediction[-2] = 0
        prediction[-5] = 0
        prediction[-6] = 0
        # press(prediction, 0.5)
        k = np.argmax(prediction)


def load(p_model_path):
    return load_model(p_model_path)


model_path = '../../models/model'
weights_path = '../../models/weights'
train_and_save_model(model_path, weights_path)

# model = load_model(model_path)
#
# # preds = model.predict(test_x[:4])
# # print(test_y[:4])
# # print(preds)
# # loss, acc = model.evaluate(test_x, test_y)
# # print('Loss: {} Acc: {}'.format(loss, acc))
#
# x = train_x
# y = train_y
# preds = model.predict(x)
# results = []
# for i in range(0, 7):
#     results.append(0)
# for predicted, expected in zip(preds, y):
#     predicted_6 = set(get_best_6_indices(predicted))
#     # predicted_6 = []
#     # for i in range(6):
#     #     predicted_6.append(randint(0, 49))
#     # predicted_6 = set(predicted_6)
#     expected_6 = set(get_best_6_indices(expected))
#     results[in_common(predicted_6, expected_6)] += 1
#
# total = len(preds)
# for i in range(0, 7):
#     print("Chance to match " + str(i) +
#           " numbers: " + str(results[i]) + "/" + str(total)
#           + " (" + str(results[i]/total) + " )")
