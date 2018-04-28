from keras.optimizers import Adam, SGD
from keras.datasets import cifar10
import numpy as np
from PIL import Image
import argparse
import math
import coremltools
import os
from cifar10_bigan_models import *
import matplotlib.pyplot as plt


## Generates image file of combined generator images

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


## Train models

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train[:, :, :, None]

X_train = X_train.reshape(X_train.shape[0], 32, 32, -1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, -1)

# Instantiate models and print no. of parameters
d = discriminator_model()
print("No. of discriminator parameters: " + str(d.count_params()))
g = generator_model()
print("No. of generator parameters: " + str(g.count_params()))
e = encoder_model()
print("No. of encoder parameters: " + str(e.count_params()))
d_on_g = generator_containing_discriminator(g, d)
d_on_e = encoder_containing_discriminator(e, d)

# Specify optimizer for each model
d_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.001)
g_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.001)
e_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.001)


# Compile models
g.compile(loss='binary_crossentropy', optimizer=g_optim)
d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
e.compile(loss='binary_crossentropy', optimizer=e_optim)
d_on_e.compile(loss='binary_crossentropy', optimizer=e_optim)
d.trainable = True
d.compile(loss='binary_crossentropy', optimizer=d_optim)


# Define training hyperparameters
batch_size = 100
epochs = 300

# Compute number of batches per epoch
num_batches = int(X_train.shape[0] / batch_size)

# Define arrays to hold progression of g and d loss
d_epoch_loss_trajectory = np.zeros(epochs)
g_epoch_loss_trajectory = np.zeros(epochs)
e_epoch_loss_trajectory = np.zeros(epochs)


# Print training details
print("Number of epochs: " + str(epochs))
print("Number of batches per epoch: " + str(num_batches))


# Train for specified number of epochs
for epoch in range(epochs):

    # Print current epoch number
    print("\nEpoch: " + str(epoch + 1) + "/" + str(epochs))

    # Set epoch losses to zero
    d_epoch_loss_sum = 0
    g_epoch_loss_sum = 0
    e_epoch_loss_sum = 0

    # Train for computed number of batches for each epoch
    for batch in range(num_batches):

        # Print current epoch and batch number
        print(
        "\nEpoch: " + str(epoch + 1) + "/" + str(epochs) + ", " + "Batch: " + str(batch + 1) + "/" + str(num_batches))

        # Generate noise input for generator and generate images from noise vector
        #noise = np.random.uniform(-1, 1, size=(batch_size, 100))
        noise = np.random.uniform(-1, 1, (batch_size, 1, 1, 64))
        #noise = np.random.lognormal(mean=0, sigma=1, size=(batch_size, 100))
        generated_images = g.predict(noise, verbose=0)

        # Create images every 20 batches
        if batch % 500 == 0:
            image = combine_images(generated_images)
            image = image * 127.5 + 127.5
            Image.fromarray(image.astype(np.uint8)).save(
                str(epoch) + "_" + str(batch) + ".png")

        # Take batch from training set
        image_batch = X_train[batch * batch_size : (batch + 1) * batch_size]

        # Encode chosen batch of images
        encoded_images = e.predict(image_batch, verbose=0)

        # Combine both types of images and code, and label accordingly
        X = np.concatenate((image_batch, generated_images))
        z = np.concatenate((encoded_images, noise))
        y = [1] * batch_size + [0] * batch_size

        # Train discriminator on batch of encoder and generator pairs
        d_loss = d.train_on_batch([X,z], y)
        print("Discriminator loss: " + str(d_loss))
        d_epoch_loss_sum += d_loss

        # Set d to be non-trainable
        d.trainable = False

        # Train generator on newly generated images
        #noise = np.random.uniform(-1, 1, (batch_size, 100))
        # noise = np.random.lognormal(mean=0, sigma=1, size=(batch_size, 100))
        noise = np.random.uniform(-1, 1, (batch_size, 1, 1, 64))
        g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
        print("Generator loss: " + str(g_loss))
        g_epoch_loss_sum += g_loss

        # Train encoder on generated images
        # !!! (Possibly might have to train on new noise vector
        e_loss = d_on_e.train_on_batch(g.predict(noise, verbose=0), [1] * batch_size)
        print("Encoder loss: " + str(e_loss))
        e_epoch_loss_sum += e_loss

        # Unfreeze discriminator parameters
        d.trainable = True

        # Save model weights every 10 batches
        if batch % 10 == 9:
            g.save_weights('cifar10_bigan_generator.h5', True)
            e.save_weights('cifar10_bigan_encoder.h5', True)
            d.save_weights('cifar10_bigan_discriminator.h5', True)

    # Average losses over all batches in epoch
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches
    e_epoch_loss_trajectory[epoch] = e_epoch_loss_sum / num_batches
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches


# Plot epoch loss curves for d, g and e
epoch_numbers = np.arange(epochs) + 1
plt.figure(1)
plt.plot(epoch_numbers, d_epoch_loss_trajectory, 'b-', epoch_numbers, g_epoch_loss_trajectory, 'g-',
         epoch_numbers, e_epoch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator', 'Encoder'])
plt.xlabel('Epoch Number')
plt.ylabel('Average Minibatch Loss')
plt.show()