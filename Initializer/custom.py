import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyCustomInitializer(keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Generate random values from a normal distribution
        return tf.random.normal(shape, mean=0., stddev=0.1, dtype=dtype)

model_custom = keras.Sequential([
    layers.Dense(
        units=64,
        activation='relu',
        kernel_initializer=MyCustomInitializer(),
        input_shape=(784,)
    ),
    layers.Dense(
        units=10,
        activation='softmax'
    )
])

model_custom.summary()
print(f"\nInitializer of first custom layer: {model_custom.layers[0].kernel_initializer.__class__.__name__}")