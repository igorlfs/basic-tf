import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers.serialization import preprocessing_normalization
from tensorflow import keras

# from keras.api._v2 import keras as KerasAPI

# |%%--%%| <H8YSZbOwpL|Iln9jli3nS>

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(precision=3, suppress=True)

# |%%--%%| <Iln9jli3nS|bNcghS0v12>

# https://archive.ics.uci.edu/ml/datasets/Auto+MPG
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

dataset = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)

# |%%--%%| <bNcghS0v12|MkwRq7GSPy>

dataset.tail()

# |%%--%%| <MkwRq7GSPy|dUj17pPXa2>

# clean data
dataset = dataset.dropna()

# convert categorical 'Origin' data into one-hot data
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1
dataset["Europe"] = (origin == 2) * 1
dataset["Japan"] = (origin == 3) * 1

# |%%--%%| <dUj17pPXa2|ilsSyCORNR>

dataset.tail()

# |%%--%%| <ilsSyCORNR|zcslCTDnZ2>

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print(dataset.shape, train_dataset.shape, test_dataset.shape)
train_dataset.describe().transpose()

# |%%--%%| <zcslCTDnZ2|IbgNich6Ik>

# split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")

# |%%--%%| <IbgNich6Ik|znZzdMc8ff>


def plot(feature, x=None, y=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(train_features[feature], train_labels, label="Data")
    if x is not None and y is not None:
        plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel(feature)
    plt.ylabel("MPG")
    plt.legend()


# |%%--%%| <znZzdMc8ff|J9rINlls8U>

plot("Horsepower")

# |%%--%%| <J9rINlls8U|NZNB72iQxj>

plot("Weight")

# |%%--%%| <NZNB72iQxj|tfSPSc5MDz>


# Normalize
print(train_dataset.describe().transpose()[["mean", "std"]])

# Normalization
normalizer = preprocessing_normalization.Normalization()

# adapt to the data
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# |%%--%%| <tfSPSc5MDz|s3SVMfPtTm>


first = np.array(train_features[:1])
print("First example:", first)
print("Normalized:", normalizer(first).numpy())

# |%%--%%| <s3SVMfPtTm|789P69DRIR>

# Regression
# 1. Normalize the input horsepower
# 2. Apply a linear transformation (y = m*x+b) to produce 1 output using layers.Dense

feature = "Horsepower"
single_feature = np.array(train_features[feature])
print(single_feature.shape, train_features.shape)

# Normalization
single_feature_normalizer = preprocessing_normalization.Normalization(
    input_shape=[
        1,
    ],
    axis=None,
)

# adapt to the data
single_feature_normalizer.adapt(single_feature)

# |%%--%%| <789P69DRIR|x24AgBOcrP>

# Sequential model
single_feature_model = keras.models.Sequential(
    [single_feature_normalizer, layers.Dense(units=1)]  # Linear Model
)

# |%%--%%| <x24AgBOcrP|QVKuEPUrcx>

single_feature_model.summary()

# |%%--%%| <QVKuEPUrcx|Hpm0sYsPy8>

# loss and optimizer
loss = keras.losses.MeanAbsoluteError()
optim = keras.optimizers.Adam(lr=0.1)

single_feature_model.compile(optimizer=optim, loss=loss)

# |%%--%%| <Hpm0sYsPy8|DewW7lMvk8>

history = single_feature_model.fit(
    train_features[feature],
    train_labels,
    epochs=100,
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split=0.2,
)


# |%%--%%| <DewW7lMvk8|Gg7Pc2ziT1>


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 25])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)


plot_loss(history)

# |%%--%%| <Gg7Pc2ziT1|wj8U6lgdA6>

single_feature_model.evaluate(test_features[feature], test_labels, verbose=1)

# |%%--%%| <wj8U6lgdA6|3UetBtRHud>

# predict and plot
range_min = np.min(test_features[feature]) - 10
range_max = np.max(test_features[feature]) + 10
x = tf.linspace(range_min, range_max, 200)
y = single_feature_model.predict(x)

plot(feature, x, y)

# |%%--%%| <3UetBtRHud|aRKiBhOilB>

# DNN
dnn_model = keras.Sequential(
    [
        single_feature_normalizer,
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)

# |%%--%%| <aRKiBhOilB|SCeQB2RTnc>

dnn_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.001))
dnn_model.summary()

# |%%--%%| <SCeQB2RTnc|iCJtzfUzzj>

dnn_model.fit(
    train_features[feature], train_labels, validation_split=0.2, verbose=1, epochs=100
)
# |%%--%%| <iCJtzfUzzj|5Gjr5mAJfv>

dnn_model.evaluate(test_features[feature], test_labels, verbose=1)

# |%%--%%| <5Gjr5mAJfv|Ln676KDCJT>

# predict and plot
x = tf.linspace(range_min, range_max, 200)
y = dnn_model.predict(x)

plot(feature, x, y)


# |%%--%%| <Ln676KDCJT|ANCDxzhUyz>

# multiple inputs
linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss=loss)

# |%%--%%| <ANCDxzhUyz|CjfYxBfeAT>

linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split=0.2,
)

# |%%--%%| <CjfYxBfeAT|YoZvUKLwQn>

linear_model.evaluate(test_features, test_labels, verbose=1)
