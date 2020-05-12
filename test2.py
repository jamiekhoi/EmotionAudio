import tensorflow as tf
batch_size = 4
steps_number = 20
hidden_units = 100
keep_prob = 0.5
dim = tf.zeros([batch_size, hidden_units])
input_data = tf.keras.layers.Input(shape=(steps_number,1), batch_size=batch_size)
hidden_1, state_h, state_c = tf.keras.layers.LSTM(
    units=hidden_units, stateful=True, dropout=keep_prob,
    return_state=True, return_sequences=True)\
    (input_data, initial_state=[dim, dim], training=True)
print(hidden_1.get_shape().as_list)
hideen_2 = tf.keras.layers.LSTM(
    units=hidden_units, stateful=True, dropout=keep_prob,
    return_state=False)\
    (hidden_1, initial_state=[state_h, state_c], training=True)
hidden3 = tf.keras.layers.Dense(10, activation='relu')(hidden_1)
output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
model = tf.keras.models.Model(input_data, output)