from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import coremltools
import os
import matplotlib.pyplot as plt


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def generate(batch_size, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator.h5')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator.h5')
        noise = np.random.uniform(-1, 1, (batch_size * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")



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



(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train[:, :, :, None]
X_test = X_test[:, :, :, None]
d = discriminator_model()
g = generator_model()
d_on_g = generator_containing_discriminator(g, d)
d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g.compile(loss='binary_crossentropy', optimizer="SGD")
d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
d.trainable = True
d.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])

epochs = 100
batch_size = 128

# Define arrays to hold progression of discriminator and bigan losses
d_epoch_loss_trajectory = np.zeros(epochs)
g_epoch_loss_trajectory = np.zeros(epochs)
d_acc_trajectory = np.zeros(epochs)

num_batches = int(X_train.shape[0] / batch_size)


for epoch in range(epochs):

    print("Epoch is", epoch)
    print("Number of batches", int(X_train.shape[0] / batch_size))

    g_epoch_loss = 0
    d_epoch_loss = 0
    d_acc = 0

    for index in range(num_batches):
        noise = np.random.uniform(-1, 1, size=(batch_size, 100))
        image_batch = X_train[index * batch_size:(index + 1) * batch_size]
        generated_images = g.predict(noise, verbose=0)
        if index % 20 == 0:
            image = combine_images(generated_images)
            image = image * 127.5 + 127.5
            Image.fromarray(image.astype(np.uint8)).save("Images/" + str(epoch) + "_" + str(index) + ".png")
        X = np.concatenate((image_batch, generated_images))
        y = [1] * batch_size + [0] * batch_size
        d_loss = d.train_on_batch(X, y)

        d_epoch_loss += d_loss[0]
        d_acc += d_loss[1]

        noise = np.random.uniform(-1, 1, (batch_size, 100))
        #noise = np.random.lognormal(mean=0, sigma=1, size=(batch_size, 100))
        d.trainable = False
        g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
        d.trainable = True
        g_epoch_loss += g_loss

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch+1, index, num_batches,
                                                                                      d_loss[0], 100 * d_loss[1],
                                                                                      g_loss))

        if index % 10 == 9:
            g.save_weights('generator.h5', True)
            d.save_weights('discriminator.h5', True)

    # Record epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss / num_batches
    d_acc_trajectory[epoch] = 100 * (d_acc / num_batches)



# Plot epoch loss curves
plt.figure()

epoch_numbers = np.arange(epochs) + 1

plt.plot(epoch_numbers, d_epoch_loss_trajectory, 'b-', epoch_numbers, g_epoch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator'], loc='upper right')
plt.xlabel('Epoch Number')
plt.ylabel('Average Minibatch Loss')

plt.savefig('Images/mnist_gan_epochloss.png')


# Plot discriminator accuracy over epochs
plt.figure()

plt.plot(epoch_numbers, d_acc_trajectory)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')

plt.savefig('Images/mnist_gan_discriminator_acc.png')
