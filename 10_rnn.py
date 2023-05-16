import os

from keras import layers
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# |%%--%%| <p9NP0C1YaV|d0M7aw5x5T>

mnist = keras.datasets.mnist

# |%%--%%| <d0M7aw5x5T|eu6WJ9ysxv>

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

# |%%--%%| <eu6WJ9ysxv|8z1BUyPszV>

# model
model = keras.models.Sequential()
model.add(keras.Input(shape=(28, 28)))
model.add(layers.SimpleRNN(128, return_sequences=False, activation="relu"))
model.add(layers.Dense(10))

# |%%--%%| <8z1BUyPszV|SdeCydaPNe>

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = keras.metrics.SparseCategoricalAccuracy()

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# |%%--%%| <SdeCydaPNe|cdIFfcBbY0>

# training
batch_size = 64
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

# |%%--%%| <cdIFfcBbY0|v730mxCl59>

# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size)

# |%%--%%| <v730mxCl59|z8cLfk3lpL>
