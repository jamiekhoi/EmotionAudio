import tensorflow as tf
import numpy as np
from modfilescapcnn import *
from sklearn.model_selection import train_test_split
from feature_builder import  get_training_data_conv

checkpoint_path = "./my_capsule_network_emotion"
batch_size = 50
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 13 * 5  # 1152 primary capsules
caps1_n_dims = 8

caps2_n_caps = 8
caps2_n_dims = 16

# Get data
X_, y_ = get_training_data_conv()
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2)  # , random_state = 0)

n_iterations_test = len(X_test) // batch_size

# Model
X, y, mask_with_labels = model_inputs()
conv2 = build_cnn_layers(X, caps1_n_maps, caps1_n_dims)

caps2_output_round_2 = build_capsnet_layers(conv2, caps1_n_caps, caps1_n_dims, X)
caps2_output = caps2_output_round_2

y_pred = get_y_pred(caps2_output)
margin_loss = get_margin_loss(y, caps2_output, caps2_n_caps)

##
decoder_input = get_decoder_input(mask_with_labels, y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims)

n_output = 39 * 13
decoder_output = get_decoder_output(decoder_input, n_output)

reconstruction_loss = get_reconstruction_loss(X, n_output, decoder_output)


alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

## Accuracy
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        #X_batch, y_batch = mnist.test.next_batch(batch_size)
        X_batch = X_test[(iteration-1)*batch_size: iteration*batch_size]
        y_batch = y_test[(iteration-1)*batch_size: iteration*batch_size]
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={
                           X: X_batch, #need to fix these shapes
                           y: np.argmax(y_batch, axis=1) # one-hot to index
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
