import numpy as np
import tensorflow as tf
from gym.envs.box2d.car_racing import CarRacing
from matplotlib import pylab
from pyglet.window import key
from tensorflow import keras
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from scipy.misc import imresize
from feature_extractor import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import png
import gzip
import os
import pickle
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from random import randint
from tensorflow import keras
from scipy import misc
from time import sleep

a = np.array([0.0, 0.0, 0.0])


def key_press(k, mod):
    global restart
    if k == 0xff0d:
        restart = True
    if k == key.LEFT:
        a[0] = -1.0
    if k == key.RIGHT:
        a[0] = +1.0
    if k == key.UP:
        a[1] = +1.0
    if k == key.DOWN:
        a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0


def get_action_list():
    action_list = [np.asarray([0.0, 0.1, 0.5]), np.asarray([0.0, 1.0, 0.0]), np.asarray([-1.0, 1.0, 0.0]),
                   np.asarray([1.0, 1.0, 0.0]), np.asarray([-1.0, 0.3, 0.0]), np.asarray([1.0, 0.3, 0.0])]
    return action_list


def user_play():
    global restart
    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            s, reward, done, info = env.step(a)
            image = np.asanyarray(s)

            # total_reward += reward
            # if steps == 50 or done:
                # features = get_features(image)
                # print(features)
                # print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                # print("step {} total_reward {:+0.2f} reward {:+0.2f}".format(steps, total_reward, reward))

                # image = get_replacement_image(image)
                # image = simplify_to_2d(image)

                # png.from_array(image, 'RGB').save('plot.png')

            steps += 1
            env.render()
            if done or restart:
                break


def do_step(env, best_action, images):
    print("Executing best action: " + str(best_action))
    global a
    a = best_action
    s, reward, done, info = env.step(a)
    image = np.asanyarray(s)
    image = get_features(image)
    images.append(image)
    return s, reward, done, info


def add_new_image(images, image):
    for i in range(len(images)-1):
        images[i] = images[i+1]
    images[len(images)-1] = image


def train(model_path='model/weights', memory_size=5, learning_rate=0.8, load_weights=False):
    global restart
    global a

    actions = np.asarray(get_action_list())
    max_env_total_reward = 0
    print("Initializing model...")
    input_size = memory_size * 35 + 3
    print('Input size: ' + str(input_size))

    # l_model = keras.Sequential([
    #     keras.layers.Dense(input_size),
    #     # keras.layers.Dense(150, activation=tf.nn.relu),
    #     keras.layers.Dense(1, activation=tf.nn.softmax)
    # ])

    l_model = Sequential()
    l_model.add(Dense(1000, input_shape=(input_size,), activation='sigmoid',
                      kernel_initializer='lecun_normal', kernel_regularizer=l2(0.01)))
    # l_model.add(Dropout(0.3))
    # l_model.add(Dense(125, activation='sigmoid'))
    l_model.add(Dropout(0.1))
    l_model.add(Dense(1, activation='tanh'))

    if load_weights:
        l_model.load_weights(model_path)

    l_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])  # optimizer = 'rmsprop'

    print("Done.")

    print("Initializing environment...")
    env = CarRacing()
    env.render()

    # we can also intervene and move the car.
    # This might ruin the learning because this changes the last action
    # To avoid this either comment the next 2 lines, either don't touch the arrow keys
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


    print("Starting playing...")
    while True:
        env.reset()
        images = []

        total_reward = 0.0
        steps = 0
        restart = False

        a = np.array([0.0, 0.0, 0.0])

        env_total_reward = 0
        print("Starting new race...")
        while True:
            far_on_grass = False
            if steps >= memory_size:
                # 1. predict Q(state, a) for each action

                # compute an numpy array arr which contains (state, a) for each action a
                state = images[-memory_size:]
                state = np.asarray(state)
                state = state.flatten()
                previous_state = state
                arr = []
                for action in actions:
                    arr.append(np.concatenate((state, action)))

                arr = np.asarray(arr)

                # prediction will be an numpy array containing predicted rewards Q(state, a) for each a
                prediction = l_model.predict(arr)

                # 2. take action a where Q(state, a) is largest
                #    receive reward R(state, a, state')

                # best_action_index = np.argmax(prediction)
                # best_action_index = np.random.choice(len(actions))

                candidate_indices = []
                max_prediction = 0
                for p in prediction:
                    if p > max_prediction:
                        max_prediction = p
                for i in range(len(prediction)):
                    if abs(max_prediction - prediction[i]) < 0.2:
                        candidate_indices.append(i)
                best_action_index = candidate_indices[np.random.choice(len(candidate_indices))]

                best_action = actions[best_action_index]

                previous_action = best_action
                # print("Taking best action: " + str(best_action))

                s, reward, done, info = do_step(env, best_action, images)
                image = np.asanyarray(s)
                env_total_reward += reward
                reward, far_on_grass = get_reward(image, best_action)
                # 3. Compute target value:
                #   3.1. Predict all Q(s', a') for every a' available in state s' using neural network

                #       compute an numpy array arr which contains (state', a') for each action a'
                #       prediction = l_model.predict(arr)
                #       prediction will be an numpy array containing predicted rewards Q(state', a') for each a'

                if steps % 100 == 0:
                    reduced_image = simplify_to_2d(image)
                    print("debug")
                state = images[-memory_size:]
                state = np.asarray(state)
                state = state.flatten()
                arr = []
                for action in actions:
                    arr.append(np.concatenate((state, action)))
                arr = np.asarray(arr)
                # arr = (np.expand_dims(arr, 0))

                prediction = l_model.predict(arr)

                #   3.2. Compute max Q(s', a') for the new state s', for every action a' in s'

                q_max = np.max(prediction)

                #   3.3. Compute y(sa) = R(s, a, s') + gamma * max(Q(s', a'))

                max_reward = 1
                max_label_value = 1.9
                y_sa = (reward/max_reward + learning_rate * q_max) / max_label_value

                #  y_sa = q_max * learning_rate
                print(" reward: " + str(reward))

                # 4. Train the neural network using:
                #   input: the previous state (s), the previous action (a)
                #   output: y(sa)

                train_input = np.concatenate((previous_state, previous_action))
                train_input = (np.expand_dims(train_input, 0))
                y_sa = (np.expand_dims(y_sa, 0))
                l_model.train_on_batch(train_input, y_sa)

            else:
                s, reward, done, info = do_step(env, a, images)

            total_reward += reward
            if steps % 50 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f} reward {:+0.2f}".format(steps, total_reward, reward))

            steps += 1
            env.render()
            if done or restart or far_on_grass:
                l_model.save_weights(model_path)
                if env_total_reward > max_env_total_reward:
                    max_env_total_reward = env_total_reward
                print("Best gym score: " + str(max_env_total_reward))
                break


def race(model_path='model/weights', memory_size=5):
    global restart

    print("Initializing model...")
    input_size = memory_size * 35 + 3
    print('Input size: ' + str(input_size))

    # l_model = keras.Sequential([
    #     keras.layers.Dense(input_size),
    #     # keras.layers.Dense(150, activation=tf.nn.relu),
    #     keras.layers.Dense(1, activation=tf.nn.softmax)
    # ])

    l_model = Sequential()
    l_model.add(Dense(1000, input_shape=(input_size,), activation='sigmoid',
                      kernel_initializer='lecun_normal', kernel_regularizer=l2(0.01)))
    # l_model.add(Dropout(0.3))
    # l_model.add(Dense(125, activation='sigmoid'))
    l_model.add(Dropout(0.1))
    l_model.add(Dense(1, activation='tanh'))

    l_model.load_weights(model_path)
    l_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])  # optimizer = 'rmsprop'

    print("Done.")

    print("Initializing environment...")
    env = CarRacing()
    env.render()

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    actions = np.asarray(get_action_list())

    print("Starting playing...")
    while True:
        env.reset()
        images = []

        total_reward = 0.0
        steps = 0
        restart = False

        print("Starting new race...")
        while True:
            far_on_grass = False
            if steps >= memory_size:
                state = images[-memory_size:]
                state = np.asarray(state)
                state = state.flatten()
                arr = []
                for action in actions:
                    arr.append(np.concatenate((state, action)))

                arr = np.asarray(arr)

                prediction = l_model.predict(arr)

                candidate_indices = []
                max_prediction = 0
                for p in prediction:
                    if p > max_prediction:
                        max_prediction = p
                for i in range(len(prediction)):
                    if abs(max_prediction - prediction[i]) < 0.2:
                        candidate_indices.append(i)
                best_action_index = candidate_indices[np.random.choice(len(candidate_indices))]
                best_action = actions[best_action_index]

                s, reward, done, info = do_step(env, best_action, images)
            else:
                s, reward, done, info = do_step(env, a, images)

            total_reward += reward
            if steps % 50 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f} reward {:+0.2f}".format(steps, total_reward, reward))

            steps += 1
            env.render()
            if done or restart or steps > 1000 or far_on_grass:
                break


# https://github.com/tensorflow/models/issues/1993
# their suggestion is to reduce the batch_size
train(memory_size=20, load_weights=True)
# /watch?v=pTYOyl8To7g
# user_play()
# de incercat sa pun actiunea la output
