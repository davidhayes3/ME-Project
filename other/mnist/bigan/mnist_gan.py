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


def train(batch_size, digit=None):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if digit is not None:
        indices = [i for i, y in enumerate(y_train) if y == digit]
        X_train = X_train[indices]
        y_train = y_train[indices]
        indices = [i for i, y in enumerate(y_test) if y == digit]
        X_test = X_test[indices]
        y_test = y_test

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    d = discriminator_model()
    g = generator_model()
    print(g.count_params())
    print(d.count_params())
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / batch_size))
        for index in range(int(X_train.shape[0] / batch_size)):
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            #noise = np.random.lognormal(mean=0, sigma=1, size=(batch_size, 100))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch) + "_" + str(index) + ".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            #noise = np.random.lognormal(mean=0, sigma=1, size=(batch_size, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator.h5', True)
                d.save_weights('discriminator.h5', True)


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


def create_coreml(filepath):
    generator = generator_model()
    if filepath is not None:
        generator.load_weights(filepath)

        # export model to coreml
        coreml_model = coremltools.converters.keras.convert(generator, input_names=['latent_space'], output_names=['digit_image'])
        coreml_model.author = 'CS-UCD'
        coreml_model.license = 'MIT'
        coreml_model.short_description = 'GAN MNIST'
        coreml_model.input_description['latent_space'] = 'array of 100 uniformly distributed numbers in range [-1, 1]'
        coreml_model.output_description['digit_image'] = '28x28 8-bit luminance'
        coreml_model.save("{}.mlmodel".format(os.path.splitext(filepath)[0]))
        print(coreml_model)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--digit", type=int, default=8)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size, digit=args.digit)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
    elif args.mode == "coreml":
        create_coreml(filepath=args.file_path)

train(128)
