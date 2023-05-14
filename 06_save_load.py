import os

from tensorflow import keras

# |%%--%%| <U09HMb7glA|7JP4KsJ6gA>

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# |%%--%%| <7JP4KsJ6gA|bdevnbbblQ>


mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
    ]
)
print(model.summary())

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=True, verbose=2)

# evaluate
model.evaluate(x_test, y_test, verbose=2)

# |%%--%%| <bdevnbbblQ|9wpDwfyl7P>

# 1) Save whole model
# SavedModel, HDF5
model.save("nn.h5")
model.save("neural_net")

# |%%--%%| <9wpDwfyl7P|xuBNcM44NU>

new_model = keras.models.load_model("../../../code/git/tf/nn.h5")
new_model.evaluate(x_test, y_test, verbose=2)

# |%%--%%| <xuBNcM44NU|enkc3rukgr>

# 2) Save only weights
model.save_weights("nn_weights.h5")
model.load_weights("nn_weights.h5")

# |%%--%%| <enkc3rukgr|267yh9xxyK>

# 3) save only architecture, to_json
json_string = model.to_json()
with open("nn_model", "w") as f:
    f.write(json_string)
with open("nn_model") as f:
    loaded_json_string = f.read()

new_model_plus = keras.models.model_from_json(loaded_json_string)
