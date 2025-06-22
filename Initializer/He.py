import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("\n--- Model with He Normal Initialization ---")
model_he = keras.Sequential([
    layers.Dense(
        units=64,
        activation='relu', # ReLU is common, so He initialization is preferred
        kernel_initializer='he_normal', # Explicitly set He normal
        input_shape=(784,)
    ),
    layers.Dense(
        units=32,
        activation='relu',
        kernel_initializer='he_uniform' # He uniform
    ),
    layers.Dense(
        units=10,
        activation='softmax'
    )
])

model_he.summary()

print(f"\nInitializer of first He layer: {model_he.layers[0].kernel_initializer.__class__.__name__}")
print(f"Initializer of second He layer: {model_he.layers[1].kernel_initializer.__class__.__name__}")