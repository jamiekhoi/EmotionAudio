import tensorflow as tf
from math_funcs import squash, safe_norm
import numpy as np

def model_inputs():
    """
    Create the model inputs
    """
    image_width = 39
    image_height = 13
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name="X")
    mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                   name="mask_with_labels")
    y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

    return X, y, mask_with_labels


def build_cnn_layers(X, caps1_n_maps, caps1_n_dims):
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
    return conv2

def build_capsnet_layers(conv2, caps1_n_caps, caps1_n_dims, X):
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
    return caps2_output_round_2

def get_y_pred(caps2_output):
    y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
    y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
    y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")
    return y_pred


def get_margin_loss(y, caps2_output, caps2_n_caps):
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
    return margin_loss

def get_decoder_input(mask_with_labels, y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims):
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
    return decoder_input

def get_decoder_output(decoder_input, n_output):
    n_hidden1 = 512
    n_hidden2 = 1024

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
        return decoder_output

def get_reconstruction_loss(X, n_output, decoder_output):
    X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
    squared_difference = tf.square(X_flat - decoder_output,
                                   name="squared_difference")
    reconstruction_loss = tf.reduce_mean(squared_difference,
                                         name="reconstruction_loss")
    return reconstruction_loss
