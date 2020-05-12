#!/usr/bin/env python
# coding: utf-8

import os

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from math_funcs import squash, safe_norm

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tfconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=tfconfig)

import os
os.chdir('..')

tf.reset_default_graph()

np.random.seed(42)
tf.set_random_seed(42)

image_width = 39
image_height = 13
X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name="X")

# # Primary Capsules
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 13 * 5  # 1152 primary capsules
caps1_n_dims = 8

# To compute their outputs, we first apply two regular convolutional layers:
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "same",  # "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims,  # 256 convolutional filters
    "kernel_size": 9,
    "strides": 3,
    "padding": "same",  # "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")

caps1_output = squash(caps1_raw, name="caps1_output")

caps2_n_caps = 8  # 10
caps2_n_dims = 16

init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")
print(W_tiled.shape)
print(caps1_output_tiled.shape)
exit()
# ## Routing by agreement
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

# ### Round 1
routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")

# ### Round 2
caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

# The rest of round 2 is the same as in round 1:
routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2

y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

# # Margin loss
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 8),
                           name="present_error")

absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 8),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

# # Reconstruction
# ## Mask
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                 lambda: y,  # if True
                                 lambda: y_pred,  # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")

caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

n_hidden1 = 512
n_hidden2 = 1024
n_output = 39 * 13
with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")

# ## Reconstruction Loss
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                     name="reconstruction_loss")

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
import pandas as pd
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from feature_builder import build_rand_feat
from cfg import AudioConfig

MAIN_PATH = 'Processed/RAVDESSspeech/'
df = pd.read_csv(MAIN_PATH + 'ravdessSpeech.csv')
df.set_index('Filename', inplace=True)

# Add length column for each file
for f in df.index:
    rate, signal = wavfile.read(MAIN_PATH + 'clean/' + f)
    df.at[f, 'length'] = signal.shape[0] / rate

classes = list(np.unique(df.Emotion))
class_dist = df.groupby(['Emotion'])['length'].mean()
n_samples = int(df['length'].sum() / 0.1)  # change this?
prob_dist = class_dist / class_dist.sum()
audio_config = AudioConfig(data_save_path=MAIN_PATH, mode='caps')

if audio_config.mode == 'caps':
    X_, y_ = build_rand_feat(audio_config, n_samples, classes, class_dist, prob_dist, df, MAIN_PATH)
    y_flat = np.argmax(y_, axis=1)  # one-hot to index
    input_shape = (X_.shape[1], X_.shape[2], 1)

n_epochs = 1#20
batch_size = 25#50
restore_checkpoint = False  # True

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2)  # , random_state = 0)

n_iterations_per_epoch = len(X_train) // batch_size
n_iterations_validation = len(X_test) // batch_size

best_loss_val = np.infty
checkpoint_path = "./my_capsule_network_emotion"


#tfconfig = tf.ConfigProto()
#tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#tfconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
#sess = tf.Session(config=tfconfig)
with tf.Session(config=tfconfig) as sess:
#with tf.Session() as sess:

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
n_samples = 5
sample_images = X_test[:n_samples]

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
        [caps2_output, decoder_output, y_pred],
        feed_dict={X: sample_images,
                   y: np.array([], dtype=np.int64)})

# Note: we feed `y` with an empty array, but TensorFlow will not use it, as explained earlier.
# And now let's plot the images and their labels, followed by the corresponding reconstructions and predictions:

sample_images = sample_images.reshape(-1, 39, 13)  # to get rid of last extra dim (of 1), could have also used squeeze

reconstructions = decoder_output_value.reshape([-1, 39, 13])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(np.argmax(y_test[index])))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")


# Let's create a function that will tweak each of the 16 pose parameters (dimensions) in all output vectors. Each tweaked output vector will be identical to the original output vector, except that one of its pose parameters will be incremented by a value varying from -0.5 to 0.5. By default there will be 11 steps (-0.5, -0.4, ..., +0.4, +0.5). This function will return an array of shape (_tweaked pose parameters_=16, _steps_=11, _batch size_=5, 1, 10, 16, 1):
def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
    steps = np.linspace(min, max, n_steps)  # -0.25, -0.15, ..., +0.25
    pose_parameters = np.arange(caps2_n_dims)  # 0, 1, ..., 15
    tweaks = np.zeros([caps2_n_dims, n_steps, 1, 1, 1, caps2_n_dims, 1])
    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    return tweaks + output_vectors_expanded


n_steps = 11

tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)
tweaked_vectors_reshaped = tweaked_vectors.reshape(
    [-1, 1, caps2_n_caps, caps2_n_dims, 1])

tweak_labels = np.tile(np.argmax(y_test[:n_samples], axis=1), caps2_n_dims * n_steps)

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    decoder_output_value = sess.run(
        decoder_output,
        feed_dict={caps2_output: tweaked_vectors_reshaped,
                   mask_with_labels: True,
                   y: tweak_labels})

# Let's reshape the decoder's output so we can easily iterate on the output dimension, the tweak steps, and the instances:

tweak_reconstructions = decoder_output_value.reshape(
    [caps2_n_dims, n_steps, n_samples, 39, 13])

# Lastly, let's plot all the reconstructions, for the first 3 output dimensions, for each tweaking step (column) and each digit (row):
for dim in range(3):
    print("Tweaking output dimension #{}".format(dim))
    plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
    for row in range(n_samples):
        for col in range(n_steps):
            plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
            plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
            plt.axis("off")
    plt.show()
print('end')
