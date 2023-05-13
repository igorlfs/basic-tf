import os
from keras import layers
import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras, optimizers

# |%%--%%| <Soih8GvDUF|l6E7IJXMNK>

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# |%%--%%| <l6E7IJXMNK|KtfhfNm3hA>

cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# |%%--%%| <KtfhfNm3hA|awX65f1oL8>

train_images, test_images = train_images / 255.0, test_images / 255.0

# |%%--%%| <awX65f1oL8|aYRjpJ6BrT>

print(train_images.shape)
print(train_labels.shape)

# |%%--%%| <aYRjpJ6BrT|JlpLtiqCxZ>

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def show():
    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()
        plt.set_cmap(matplotlib.colors.Colormap("binary"))
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i][0]])


show()

# |%%--%%| <JlpLtiqCxZ|3M6HbxDFXW>

model = keras.models.Sequential()
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        input_shape=(32, 32, 3),
    )
)
model.add(layers.MaxPool2D((2, 2)))
model.add(
    layers.Conv2D(
        32,
        3,
        activation="relu",
    )
)
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

print(model.summary())

# |%%--%%| <3M6HbxDFXW|jSjHYoziB0>

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# |%%--%%| <jSjHYoziB0|ZZjZdOxwoq>

batch_size = 64
epochs = 5

model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)

# |%%--%%| <ZZjZdOxwoq|MyvlUWWOcg>

model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)

# |%%--%%| <MyvlUWWOcg|AgO95kSCL4>
