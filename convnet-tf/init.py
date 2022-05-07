#!/usr/bin/env Python
# -*- coding: utf-8 -*-

# TensorFlow 1.11.0

from __future__ import absolute_import, division, print_function

import argparse, sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

class Constants(object):
    SEED = None #39
    NUM_ITERATION = 20000

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[ 1, 1, 1, 1 ], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[ 1, 2, 2, 1 ], strides=[ 1, 2, 2, 1 ], padding="SAME")

def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape, stddev=.1, seed=Constants.SEED))

def bias_variable(shape):
    return tf.Variable(initial_value=tf.constant(.1, shape=shape))

def deepnn(x):
    with tf.name_scope("reshape"):
        x_image = tf.reshape(x, [ -1, 28, 28, 1 ])

    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([ 5, 5, 1, 32 ])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([ 5, 5, 32, 64 ])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([ 7 * 7 * 64, 1024 ])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [ -1, 7 * 7 * 64 ])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([ 1024, 10 ])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

def init(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [ None, 784 ])
    y_= tf.placeholder(tf.float32, [ None, 10 ])

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope("adam_optimizer"):
        train_step = tf.train.AdamOptimizer(1E-04).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for k in range(Constants.NUM_ITERATION):
            batch = mnist.train.next_batch(50)
            if k % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    keep_prob: 1.
                    })
                print("Step[%d]: Training Accuracy: %g" % (k, train_accuracy))
            train_step.run(feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: .5
                })

        print("Testing Accuracy: %g" % accuracy.eval(feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels,
            keep_prob: 1.
            }))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./mnist/input_data/", help="Dir for storing input data")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=init, argv=[sys.argv[0]] + unparsed)
