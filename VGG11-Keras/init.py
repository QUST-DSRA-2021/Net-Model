#!/usr/bin/env Python
# -*- coding: utf-8 -*-

# TensorFlow version: 1.11.0 -> 1.13.1
# Keras version: 2.1.6 -> 2.3.1
# Python Version: 3.6.5 -> 3.7.0

__author__ = "sandyzikun"

import time
import numpy as np
import keras, keras.backend as T

from matplotlib import pyplot as plt
plt.style.use("solarized-light")

class Constants(object):
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    NUM_EPOCHES = 12
    IMG_ROWS = 28 * 2
    IMG_COLS = 28 * 2
    DIR_LOGS = "./logs-vgg16/"

(X_Train, y_Train), (X_Test, y_Test) = keras.datasets.mnist.load_data()

if T.image_data_format() == "channels_first":
    Constants.INPUT_SHAPE = (1, Constants.IMG_ROWS, Constants.IMG_COLS)
else:
    Constants.INPUT_SHAPE = (Constants.IMG_ROWS, Constants.IMG_COLS, 1)

def mnist2vgg16(x: np.ndarray) -> np.ndarray:
    res = np.zeros((x.shape[0], Constants.IMG_ROWS, Constants.IMG_COLS))
    res[ : , : : 2 , : : 2 ] += x
    res[ : , 1 : : 2 , : : 2 ] += x
    res[ : , : : 2 , 1 : : 2 ] += x
    res[ : , 1 : : 2 , 1 : : 2 ] += x
    res = res.astype(np.float32)
    res = res.reshape((x.shape[0],) + Constants.INPUT_SHAPE)
    res /= 255
    return res

def plothist(h: keras.callbacks.History, s: str, plotval: bool = True) -> None:
    plt.title("History %s of Training %s (at %s)" % (s, h.model.name, hex(id(h.model))))
    if plotval:
        plt.plot(h.history[s])
        plt.plot(h.history["val_%s" % s], "-o")
        plt.legend([s, "val_%s" % s])
    else:
        plt.plot(h.history[s], "-o")
        plt.legend([s])
    plt.xlabel("epoches")
    plt.ylabel(s)
    plt.savefig(Constants.DIR_LOGS + "%s-%s.%s.jpeg" % (h.model.name, s, time.time()))
    plt.clf()

X_Train = mnist2vgg16(X_Train)
X_Test = mnist2vgg16(X_Test)
y_Train = keras.utils.to_categorical(y_Train, Constants.NUM_CLASSES)
y_Test = keras.utils.to_categorical(y_Test, Constants.NUM_CLASSES)

print("Tensor Shape of each Image:", X_Train[ 0 , : , : , : ].shape)
print("Num of Training Samples:", X_Train.shape[0])
print("Num of Testing Samples:", X_Test.shape[0])

#vgg16 = keras.applications.VGG16(include_top=True, weights=None, input_shape=Constants.INPUT_SHAPE, classes=10)
vgg11 = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", input_shape=Constants.INPUT_SHAPE, name="block1_conv1"),
    keras.layers.MaxPool2D(name="block1_pool"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="block2_conv1"),
    keras.layers.MaxPool2D(name="block2_pool"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="block3_conv1"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="block3_conv2"),
    keras.layers.MaxPool2D(name="block3_pool"),
    keras.layers.Dropout(.2, name="block3_dropout"),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name="block4_conv1"),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name="block4_conv2"),
    keras.layers.MaxPool2D(name="block4_pool"),
    keras.layers.Dropout(.2, name="block4_dropout"),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name="block5_conv1"),
    keras.layers.Conv2D(512, 3, activation="relu", padding="same", name="block5_conv2"),
    keras.layers.MaxPool2D(name="block5_pool"),
    keras.layers.Dropout(.2, name="block5_dropout"),
    keras.layers.Flatten(name="flatten"),
    keras.layers.Dense(4096, activation="relu", name="fc1"),
    keras.layers.Dropout(.2, name="fc1dropout"),
    keras.layers.Dense(4096, activation="relu", name="fc2"),
    keras.layers.Dropout(.2, name="fc2dropout"),
    keras.layers.Dense(Constants.NUM_CLASSES, activation="softmax", name="predictions")
    ])

vgg11.compile(
        loss = keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adadelta(),
        metrics = ["accuracy"])

if __name__ == "__main__":
    time0 = time.time()
    hist = vgg11.fit(
        X_Train, y_Train,
        batch_size = Constants.BATCH_SIZE,
        epochs = Constants.NUM_EPOCHES,
        verbose=1,
        validation_data = (X_Test, y_Test))
    plothist(hist, "acc")
    plothist(hist, "loss")
    score = vgg11.evaluate(X_Test, y_Test, verbose=1)
    print("Training Time:", time.time() - time0)
    print("Testing Loss:", score[0])
    print("Testing Accuracy:", score[1])
