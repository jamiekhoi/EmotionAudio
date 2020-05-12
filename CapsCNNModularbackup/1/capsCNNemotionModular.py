#!/usr/bin/env python
# coding: utf-8

import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from math_funcs import squash, safe_norm
import pandas as pd
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from feature_builder import build_rand_feat, get_training_data_conv
from cfg import AudioConfig
np.random.seed(42)
tf.set_random_seed(42)

tf.reset_default_graph()

from modfilescapcnn import *

# # Primary Capsules
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 13 * 5  # 1152 primary capsules
caps1_n_dims = 8

caps2_n_caps = 8
caps2_n_dims = 16

batch_size = 50

X, y, mask_with_labels = model_inputs()
conv2 = build_cnn_layers(X, caps1_n_maps, caps1_n_dims)

caps2_output_round_2 = build_capsnet_layers(conv2, caps1_n_caps, caps1_n_dims, X)
caps2_output = caps2_output_round_2

y_pred = get_y_pred(caps2_output)

margin_loss = get_margin_loss(y, caps2_output, caps2_n_caps)

# # Reconstruction
# ## Mask

decoder_input = get_decoder_input(mask_with_labels, y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims)

n_output = 39 * 13
decoder_output = get_decoder_output(decoder_input, n_output)

# ## Reconstruction Loss
reconstruction_loss = get_reconstruction_loss(X, n_output, decoder_output)

# ## Final Loss
alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

# # Final Touches
# ## Accuracy
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# ## Training Operations
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

# ## Init and Saver
# And let's add the usual variable initializer, as well as a `Saver`:
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# # Training
X_, y_ = get_training_data_conv()

n_epochs = 1  # 20
#batch_size = 50
restore_checkpoint = False  # True

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2)  # , random_state = 0)

n_iterations_per_epoch = len(X_train) // batch_size
n_iterations_validation = len(X_test) // batch_size

best_loss_val = np.infty
checkpoint_path = "./my_capsule_network_emotion"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch = X_train[(iteration - 1) * batch_size: iteration * batch_size]
            y_batch = y_train[(iteration - 1) * batch_size: iteration * batch_size]
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={  # X: X_batch.reshape([-1, 28, 28, 1]), #need to fix these shapes
                    X: X_batch,  # need to fix these shapes
                    y: np.argmax(y_batch, axis=1),  # one-hot to index,
                    mask_with_labels: True
                })
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                iteration, n_iterations_per_epoch,
                iteration * 100 / n_iterations_per_epoch,
                loss_train),
                end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch = X_test[(iteration - 1) * batch_size: iteration * batch_size]
            y_batch = y_test[(iteration - 1) * batch_size: iteration * batch_size]
            loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={  # X: X_batch.reshape([-1, 28, 28, 1]), #need to fix these shapes
                    X: X_batch,  # need to fix these shapes
                    # y: y_batch
                    y: np.argmax(y_batch, axis=1)  # one-hot to index
                })
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_validation,
                iteration * 100 / n_iterations_validation),
                end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val

# # Evaluation
n_iterations_test = len(X_test) // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        # X_batch, y_batch = mnist.test.next_batch(batch_size)
        X_batch = X_test[(iteration - 1) * batch_size: iteration * batch_size]
        y_batch = y_test[(iteration - 1) * batch_size: iteration * batch_size]
        loss_test, acc_test = sess.run(
            [loss, accuracy],
            feed_dict={  # X: X_batch.reshape([-1, 28, 28, 1]),
                # y: y_batch
                X: X_batch,  # need to fix these shapes
                y: np.argmax(y_batch, axis=1)  # one-hot to index
            })
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
            iteration, n_iterations_test,
            iteration * 100 / n_iterations_test),
            end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))

    assert acc_test * 100 > 23, acc_test * 100

# # Predictions
#capsCNNemotionModularPrediction.py

# Note: we feed `y` with an empty array, but TensorFlow will not use it, as explained earlier.
# Plot the images and their labels, followed by the corresponding reconstructions and predictions:
#capsCNNemotionModularPrediction.py

# Tweak output vectors to see what their pose parameters represent
#capsCNNemotionModularPrediction.py