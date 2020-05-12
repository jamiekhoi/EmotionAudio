import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modfilescapcnn import *
from sklearn.model_selection import train_test_split
from feature_builder import  get_training_data_conv

# # Primary Capsules
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 13 * 5  # 1152 primary capsules
caps1_n_dims = 8

caps2_n_caps = 8
caps2_n_dims = 16


# Get data
X_, y_ = get_training_data_conv()
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2)  # , random_state = 0)

# Model
X, y, mask_with_labels = model_inputs()
conv2 = build_cnn_layers(X, caps1_n_maps, caps1_n_dims)

caps2_output_round_2 = build_capsnet_layers(conv2, caps1_n_caps, caps1_n_dims, X)
caps2_output = caps2_output_round_2

y_pred = get_y_pred(caps2_output)

decoder_input = get_decoder_input(mask_with_labels, y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims)

n_output = 39 * 13
decoder_output = get_decoder_output(decoder_input, n_output)

checkpoint_path = "./my_capsule_network_emotion"
saver = tf.train.Saver()

# Predictions
n_samples = 5
sample_images = X_test[:n_samples]
print(sample_images.shape)

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [caps2_output, decoder_output, y_pred],
            feed_dict={X: sample_images,
                       y: np.array([], dtype=np.int64)})


# Print Reconstructions from decoder
# Plot the images and their labels, followed by the corresponding reconstructions and predictions:
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
plt.show()

# Tweak output vectors to see what their pose parameters represent
print(caps2_output_value.shape)
#print(caps2_output_value)

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
