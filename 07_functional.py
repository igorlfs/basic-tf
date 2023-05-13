from tensorflow import keras

# |%%--%%| <yeZrkJFk2J|kjArPe7nf1>

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
    ]
)

print(model.summary())

# |%%--%%| <kjArPe7nf1|TK7xc64icM>

inputs = keras.Input(shape=(28, 28))

flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(128, activation="relu")
dense2 = keras.layers.Dense(10)
dense2_2 = keras.layers.Dense(1)

x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)
outputs2 = dense2_2(x)

model = keras.Model(inputs=inputs, outputs=[outputs, outputs2], name="functional_model")

print(model.summary())

# |%%--%%| <TK7xc64icM|tgXMBKnBIz>


# access inputs, outputs for model
# access input + output for layer
# access all layers
inputs = model.inputs
outputs = model.outputs
print(inputs)
print(outputs)

input0 = model.layers[0].input
output0 = model.layers[0].output
print(input0)
print(output0)

# |%%--%%| <tgXMBKnBIz|ARrZ7q5iSd>


# Example: Transfer Learning:
base_model = keras.applications.VGG16()

x = base_model.layers[-2].output
new_outputs = keras.layers.Dense(1)(x)

new_model = keras.Model(inputs=base_model.inputs, outputs=new_outputs)
