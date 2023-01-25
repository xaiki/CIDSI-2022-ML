import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

input_shape = [28,28,1]
x_train = tf.reshape(x_train, [x_train.shape[0]] + input_shape)
x_test = tf.reshape(x_test, [x_test.shape[0]] + input_shape)

x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

x_train /= 255
x_test /= 255

y_train = tf.reshape(y_train, [-1,1])
y_test = tf.reshape(y_test, [-1,1])

encoder = OneHotEncoder(sparse=False)
y_train = tf.convert_to_tensor(encoder.fit_transform(y_train))
y_test = tf.convert_to_tensor(encoder.fit_transform(y_test))

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(28, (3,3), input_shape=input_shape),
#     tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=input_shape,
                           filters=8, kernel_size=5, strides=1, activation='relu', kernel_initializer='variance_scaling'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(kernel_size=5, filters=16, strides=1, activation='relu', kernel_initializer='variance_scaling'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer='variance_scaling')
])

loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

h = model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_test, y_test),  batch_size=64)
model.save('model2.h5')
