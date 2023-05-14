import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.preprocessing.image import DirectoryIterator
from tensorflow import keras

# |%%--%%| <PVydPeZPje|xUKEbtzOIi>

BASE_DIR = "lego/star-wars/"
DIRS = ["train/", "val/", "test/"]
names = ["YODA", "LUKE", "R2", "MACE", "GRIEVOUS"]
tf.random.set_seed(1)

# |%%--%%| <xUKEbtzOIi|1R4eh5kUty>

if not os.path.isdir(BASE_DIR + "train/"):
    for name in names:
        for directory in DIRS:
            os.makedirs(BASE_DIR + directory + name)

# |%%--%%| <1R4eh5kUty|ZVWDW5f8Wr>

ORIG_FOLDERS = ["0001/", "0002/", "0003/", "0004/", "0005/"]
for folder_idx, folder in enumerate(ORIG_FOLDERS):
    files = os.listdir(BASE_DIR + folder)
    number_of_images = len(list(files))
    n_train = int((number_of_images * 0.6) + 0.5)
    n_valid = int((number_of_images * 0.25) + 0.5)
    n_test = number_of_images - (n_train + n_valid)
    print(number_of_images, n_train, n_valid, n_test)
    for idx, file in enumerate(files):
        file_name = BASE_DIR + folder + file
        if idx < n_train:
            shutil.move(file_name, BASE_DIR + DIRS[0] + names[folder_idx])
        elif idx < n_train + n_valid:
            shutil.move(file_name, BASE_DIR + DIRS[1] + names[folder_idx])
        else:
            shutil.move(file_name, BASE_DIR + DIRS[2] + names[folder_idx])

# |%%--%%| <ZVWDW5f8Wr|A74weiycQx>


def get_batches(
    directory: str, shuffle: bool, size: tuple[int, int] = (256, 256)
) -> DirectoryIterator:
    return train_gen.flow_from_directory(
        BASE_DIR + directory,
        target_size=size,
        class_mode="sparse",
        batch_size=4,
        shuffle=shuffle,
        color_mode="rgb",
        classes=names,
    )


# |%%--%%| <A74weiycQx|Rb11rQBgyk>

# Generate batches of tensor image data with
# optional real-time data augmentation.
# preprocessing_function
# rescale=1./255 -> [0,1]
train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
#    rotation_range=20,
#    horizontal_flip=True,
#    width_shift_range=0.2, height_shift_range=0.2,
#    shear_range=0.2, zoom_range=0.2)

valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_batches = get_batches("train", True)
val_batches = get_batches("val", False)
test_batches = get_batches("test", False)

# |%%--%%| <Rb11rQBgyk|y0MKA9H129>

train_batch = train_batches[0]
print(train_batch[0].shape)
print(train_batch[1])
test_batch = test_batches[0]
print(test_batch[0].shape)
print(test_batch[1])

# |%%--%%| <y0MKA9H129|1xBCszUBDe>


def show(batch, pred_labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        lbl = names[int(batch[1][i])]
        if pred_labels is not None:
            lbl += "/ Pred:" + names[int(pred_labels[i])]
        plt.xlabel(lbl)
    plt.show()


show(test_batch)

# |%%--%%| <1xBCszUBDe|0ODKWLgC1J>

show(train_batch)

# |%%--%%| <0ODKWLgC1J|8UpJ36vbOV>

model = keras.models.Sequential()
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        input_shape=(256, 256, 3),
    )
)
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(5))
print(model.summary())

# |%%--%%| <8UpJ36vbOV|ehQzr3tlQ7>

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# |%%--%%| <ehQzr3tlQ7|0hE9QtQOHV>

# training
epochs = 30

# callbacks
# infelizmente esse treinamento crasha meu PC
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, verbose=2
)

history = model.fit(
    train_batches,
    validation_data=val_batches,
    callbacks=[early_stopping],
    epochs=epochs,
    verbose=2,
)

# |%%--%%| <0hE9QtQOHV|G5UecnTrhh>

model.save("lego_model.h5")

# |%%--%%| <G5UecnTrhh|v6fA8alaEC>

# plot loss and acc
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="valid loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="valid acc")
plt.grid()
plt.legend()

# |%%--%%| <v6fA8alaEC|2XkbXnpcdp>

# evaluate on test data
model.evaluate(test_batches, verbose=2)

# |%%--%%| <2XkbXnpcdp|PvsSdGOFAy>

# make some predictions
predictions = model.predict(test_batches)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)

print(test_batches[0][1])
print(labels[0:4])

# |%%--%%| <PvsSdGOFAy|TmTuv0H8gE>

show(test_batches[0], labels[0:4])
# |%%--%%| <TmTuv0H8gE|xzkIbHWlCv>
r"""°°°
# Transfer Learning
°°°"""
# |%%--%%| <xzkIbHWlCv|uF6JE9rGey>
vgg_model = tf.keras.applications.vgg16.VGG16()
print(type(vgg_model))
vgg_model.summary()

# try out different ones, e.g. MobileNetV2
# tl_model = tf.keras.applications.MobileNetV2()
# print(type(tl_model))
# tl_model.summary()

# |%%--%%| <uF6JE9rGey|wQmcm7J5Un>

# convert to Sequential model, omit the last layer
# this works with VGG16 because the structure is linear
model = keras.models.Sequential()
for layer in vgg_model.layers[0:-1]:
    model.add(layer)

# |%%--%%| <wQmcm7J5Un|J037bpeEQJ>

model.summary()

# |%%--%%| <J037bpeEQJ|cHf6qouN4N>

# set trainable=False for all layers
# we don't want to train them again
for layer in model.layers:
    layer.trainable = False
model.summary()

# |%%--%%| <cHf6qouN4N|kRFhM6spLu>

# add a last classification layer for our use case with 5 classes
model.add(layers.Dense(5))

# |%%--%%| <kRFhM6spLu|3BRecyuw8i>

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# |%%--%%| <3BRecyuw8i|rENeK2JJrn>

# get the preprocessing function of this model
preprocess_input = tf.keras.applications.vgg16.preprocess_input

# |%%--%%| <rENeK2JJrn|fpNHGFCkHP>

# Generate batches of tensor image data with real-time data augmentation.

train_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
valid_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
test_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_batches = get_batches("train", True, (224, 224))
val_batches = get_batches("val", True, (224, 224))
test_batches = get_batches("test", False, (224, 224))


# |%%--%%| <fpNHGFCkHP|heqsPYFGwb>

epochs = 30

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, verbose=2
)

model.fit(
    train_batches,
    validation_data=val_batches,
    callbacks=[early_stopping],
    epochs=epochs,
    verbose=2,
)

# |%%--%%| <heqsPYFGwb|TBiwLfxNJv>

model.evaluate(test_batches, verbose=2)

# |%%--%%| <TBiwLfxNJv|BHZFjWhnBM>
