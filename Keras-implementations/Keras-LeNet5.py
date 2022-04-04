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
    IMG_ROWS = 28 + 4
    IMG_COLS = 28 + 4
    DIR_LOGS = "./logs-lenet/"

(X_Train, y_Train), (X_Test, y_Test) = keras.datasets.mnist.load_data()

if T.image_data_format() == "channels_first":
    Constants.INPUT_SHAPE = (1, Constants.IMG_ROWS, Constants.IMG_COLS)
else:
    Constants.INPUT_SHAPE = (Constants.IMG_ROWS, Constants.IMG_COLS, 1)

def mnist2lenet(x: np.ndarray) -> np.ndarray:
    res = np.zeros((x.shape[0], 32, 32))
    res[ : , 2 : (-2) , 2 : (-2) ] += x
    res = res.reshape((x.shape[0],) + Constants.INPUT_SHAPE)
    res = res.astype(np.float32)
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

X_Train = mnist2lenet(X_Train)
X_Test = mnist2lenet(X_Test)
y_Train = keras.utils.to_categorical(y_Train, Constants.NUM_CLASSES)
y_Test = keras.utils.to_categorical(y_Test, Constants.NUM_CLASSES)

print("Tensor Shape of each Image:", X_Train[ 0 , : , : , : ].shape)
print("Num of Training Samples:", X_Train.shape[0])
print("Num of Testing Samples:", X_Test.shape[0])

lenet = keras.Sequential([
    keras.layers.Conv2D(filters=6, kernel_size=5, activation="relu", input_shape=Constants.INPUT_SHAPE, name="C1"),
    keras.layers.AvgPool2D(name="SP2"),
    keras.layers.Dropout(.25),
    keras.layers.Conv2D(16, 5, activation="relu", name="C3"),
    keras.layers.AvgPool2D(name="SP4"),
    keras.layers.Conv2D(120, 5, activation="relu", name="C5"),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu", name="F6"),
    keras.layers.Dropout(.5),
    keras.layers.Dense(Constants.NUM_CLASSES, activation="softmax", name="Output"),
    ], name="LeNet-5")

lenet.compile(
        loss = keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adadelta(),
        metrics = ["accuracy"])

if __name__ == "__main__":
    time0 = time.time()
    hist = lenet.fit(
            X_Train, y_Train,
            batch_size = Constants.BATCH_SIZE,
            epochs = Constants.NUM_EPOCHES,
            verbose = 1,
            validation_data = (X_Test, y_Test))
    plothist(hist, "acc")
    plothist(hist, "loss")
    score = lenet.evaluate(X_Test, y_Test, verbose=1)
    print("Training Time:", time.time() - time0)
    print("Testing Loss:", score[0])
    print("Testing Accuracy:", score[1])
