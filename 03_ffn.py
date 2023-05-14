import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# autocompletion only works with pyright (not with pylsp)

# see:
# https://github.com/tensorflow/tensorflow/issues/56231, in particular https://stackoverflow.com/a/75346479
# https://stackoverflow.com/q/68860879
# https://github.com/microsoft/pylance-release/issues/1066


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# |%%--%%| <p9NP0C1YaV|d0M7aw5x5T>

mnist = keras.datasets.mnist

# |%%--%%| <d0M7aw5x5T|nzEYMq96gn>

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# |%%--%%| <nzEYMq96gn|4e2Pq6nKJc>

# normalize 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

# |%%--%%| <4e2Pq6nKJc|pq0e1aPwcY>

# sample plot
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train[i], cmap="gray")
plt.show()

# |%%--%%| <pq0e1aPwcY|8z1BUyPszV>

# model
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
    ]
)
print(model.summary())

# |%%--%%| <8z1BUyPszV|SdeCydaPNe>

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# |%%--%%| <SdeCydaPNe|cdIFfcBbY0>

# training
batch_size = 64
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

# |%%--%%| <cdIFfcBbY0|v730mxCl59>

# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size)

# |%%--%%| <v730mxCl59|4eTUIfEWyw>

# predictions
# probabily_model = keras.models.Sequential([model, keras.layers.Softmax()])
#
# predictions = probabily_model(x_test)
# pred0 = predictions[0]
# print(pred0)
# label0 = np.argmax(pred0)
# print(label0)

predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)
