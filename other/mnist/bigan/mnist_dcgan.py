from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np


img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

optimizer = Adam(0.0002, 0.5)


def save_imgs(gen_imgs, epoch):
    r, c = 5, 5

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    # fig.suptitle("DCGAN: Generated digits", fontsize=12)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig("Images/mnist_dcgan_%d.png" % epoch)
    plt.close()


def build_generator():
    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_shape=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model



def build_discriminator():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model



# Build and compile the discriminator
print('Discriminator')
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

# Build and compile the generator
print('Generator')
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# The generator takes noise as input and generated imgs
z = Input(shape=(100,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)



# Train models

# Load the dataset
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)


# Training hyperparameters

epochs = 100
batch_size = 32
save_interval = 5
num_batches = int(X_train.shape[0] / batch_size)
half_batch = int(batch_size / 2)


# Define arrays to hold progression of discriminator and bigan losses
d_batch_loss_trajectory = np.zeros(epochs * num_batches)
g_batch_loss_trajectory = np.zeros(epochs * num_batches)
d_epoch_loss_trajectory = np.zeros(epochs)
g_epoch_loss_trajectory = np.zeros(epochs)



for epoch in range(epochs):

    # Print current epoch number
    print("\nEpoch: " + str(epoch + 1) + "/" + str(epochs))

    # Set epoch losses to zero
    d_epoch_loss_sum = 0
    g_epoch_loss_sum = 0

    # Train on all batches
    for batch in range(num_batches):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select next batch of images from training set and encode
        imgs = X_train[batch * batch_size: (batch + 1) * batch_size]

        ## Train d on full batch

        # Sample noise and generate img
        z = np.random.normal(size=(batch_size, latent_dim))
        gen_imgs = generator.predict(z)

        # Create labels for discriminator inputs
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        ## Train d on half batch



        '''# Sample noise and generate img
        z = np.random.normal(size=(half_batch, latent_dim))
        gen_imgs = generator.predict(z)

        # Select a random half of image batch and encode
        idx = np.random.randint(0, batch_size, half_batch)
        imgs = imgs[idx]
        
        # Create labels for discriminator inputs
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))'''


        ## Train the discriminator (img -> z is valid, z -> img is fake)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        ## Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Train the generator (wants discriminator to mistake images as real)
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        g_batch_loss_trajectory[epoch * num_batches + batch] = g_loss
        g_epoch_loss_sum += g_loss

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, batch, num_batches,
            d_loss[0], 100 * d_loss[1], g_loss))


    # Get epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches


    # If at save interval => save generated image samples
    if epoch % save_interval == 0:
        noise = np.random.normal(0, 1, (25, 100))
        gen_imgs = generator.predict(noise)
        save_imgs(gen_imgs, epoch)


## Visualization

# Plot loss curves

# Plot batch loss curves for g and d
plt.figure(1)
batch_numbers = np.arange((epochs * num_batches)) + 1
plt.plot(batch_numbers, d_batch_loss_trajectory, 'b-', batch_numbers, g_batch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator'], loc='upper right')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()


# Plot epoch loss curves for g and d
plt.figure(2)
epoch_numbers = np.arange(epochs) + 1
plt.plot(epoch_numbers, d_epoch_loss_trajectory, 'b-', epoch_numbers, g_epoch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator'], loc='upper left')
plt.xlabel('Epoch Number')
plt.ylabel('Average Minibatch Loss')
plt.savefig('Images/mnist_bigan_valloss_%d_epochs_%d_bs.png' % (epochs, batch_size))
plt.show()