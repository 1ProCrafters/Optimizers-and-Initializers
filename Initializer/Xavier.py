import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow Version: {tf.__version__}")

print("\n--- Model with Xavier (Glorot) Uniform Initialization ---")
model_xavier = keras.Sequential([
    # Input layer needs input_shape for the first Dense layer
    layers.Dense(
        units=64,
        activation='tanh', # tanh is symmetric around zero, good for Xavier
        kernel_initializer='glorot_uniform', # Explicitly set Xavier uniform
        input_shape=(784,) # Example input shape (e.g., flattened MNIST image)
    ),
    layers.Dense(
        units=32,
        activation='tanh',
        kernel_initializer='glorot_normal' # Xavier normal
    ),
    layers.Dense(
        units=10,
        activation='softmax' # Output layer activation
    )
])

# Print a summary of the model to see the layers and parameter counts
model_xavier.summary()

# You can also inspect the initializer of a specific layer
print(f"\nInitializer of first Xavier layer: {model_xavier.layers[0].kernel_initializer.__class__.__name__}")
print(f"Initializer of second Xavier layer: {model_xavier.layers[1].kernel_initializer.__class__.__name__}")

