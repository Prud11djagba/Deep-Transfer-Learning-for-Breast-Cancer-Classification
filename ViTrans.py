'''An implementation of a vision transformer for breast cancer classification


The original code has been taken from:
https://keras.io/examples/vision/image_classification_with_vision_transformer/
'''

# Importing libraries

from torchvision import transforms
import glob
import random 
import numpy as np


# Copying files path in an array
datas = glob.glob(r'/home/aimsgh-02/AIMS/DATA/**/*.png', recursive = True)

print("The image's path have been recorded ")

# Define the data and targets
data = random.sample(datas, int(len(datas)))
targets = np.array([[int(img[-5])] for img in data])

print(len(data))

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# loading images using pillow
for i in range(len(data)):
    data[i] = np.array(Image.open(data[i]).resize((50, 50)))
    if i % 10000 == 0:
        print("You have {} steps left".format(len(targets)-i))

print("The images  have been loaded ")


# Creating an empty array for the dataset
dataset = np.empty((len(data), 50, 50, 3))

# Changing the data object's shape
for i, image in enumerate(data):
    dataset[i] = image
    if i % 10000 == 0:
        print("You have {} steps left".format(len(targets)-i))
# Print the shape of the dataset
print("Dataset shape:", dataset.shape)

print("The images  have been reshaped ")


# Splitting the data into train (80%) and test (20%)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataset, targets, test_size=0.2, random_state=42)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#! pip install -U tensorflow-addons

import tensorflow_addons as tfa


# Defining hyperparameters

num_classes = 2
input_shape = (50, 50, 3)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 10
num_epochs = 10
image_size = 72  # changing image's size to this
patch_size = 6  # patches size
num_patches = (image_size // patch_size) ** 2
projection_dim = 32 
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

# Image preprocessing

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Computing the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)


# Multi-layer perceptron

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Patches creation

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Plotting example for patches creation

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")
plt.savefig('image_to_patch0.png')
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
    plt.savefig('image_to_patch1.png')


# Transformer encoder

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# Vit model

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# Model training


def run_experiment(model):
    optimizer = tf.keras.optimizers.SGD(
    learning_rate= learning_rate,
    momentum=0.0,
    nesterov=False,
    weight_decay=weight_decay,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name="SGD"
)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )
    
    from sklearn.metrics import classification_report

    # ...

    # After evaluating the model on the test set
    y_pred = np.argmax(vit_classifier.predict(x_test), axis=-1)
    y_true = np.squeeze(y_test)

    # Calculate precision, recall, and F1-score
    report = classification_report(y_true, y_pred)
    print(report)
    

    # Plotting the evolution of loss and accuracy
    plt.figure(figsize=(12, 4))

    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history
import time

# Start the timer
start_time = time.time()

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

end_time = time.time()  
    
# Calculate the running time
running_time = end_time - start_time

# Print or log the running time
print(f"Running time: {running_time:.2f} seconds")
