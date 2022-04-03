#!/usr/bin/env Python
# -*- coding: utf-8 -*-

# TensorFlow version: 1.13.1
# Keras version: 2.3.1
# Python Version: 3.7.0

__author__ = "sandyzikun"

import keras, keras.backend as T

class Constants(object):
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    NUM_EPOCHES = 12
    IMG_ROWS = 28
    IMG_COLS = 28

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

X_Train = mnist2lenet(X_Train)
X_Test = mnist2lenet(X_Test)
y_Train = keras.utils.to_categorical(y_Train, Constants.NUM_CLASSES)
y_Test = keras.utils.to_categorical(y_Test, Constants.NUM_CLASSES)

print("Tensor Shape of each Image:", X_Train[ 0 , : , ; , : ].shape)
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
    ])

lenet.compile(
        loss = keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adadelta(),
        metrics = ["accuracy"])

if __name__ == "__main__":
    hist = lenet.fit() # TODO::FIXME!!
