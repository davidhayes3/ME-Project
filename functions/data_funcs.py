import numpy as np
from keras.datasets import mnist, cifar10, cifar100


# Function to rescale images
def rescale_image(image, image_range=(0,1)):
    if image_range is (0,1):
        image *= 255
    elif image_range is (-1,1):
        image = 127.5 * image + 127.5

    return image


# Function to pre-process data
def preprocess_data(data, gan=False, color=False):
    if gan is False:
        data = data.astype(np.float32) / 255.
    elif gan is True:
        data = (data.astype(np.float32) - 127.5) / 127.5
    else:
        print('Incorrect range of values requested')

    if color is not True:
        data = np.expand_dims(data, axis=3)

    return data


# Function to load and pre-process MNIST dataset
def get_mnist(gan=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = preprocess_data(X_train, gan)
    X_test = preprocess_data(X_test, gan)

    return (X_train, y_train), (X_test, y_test)


# Function to load and pre-process CIFAR10 dataset
def get_cifar10(gan=False):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = preprocess_data(X_train, gan, color=True)
    X_test = preprocess_data(X_test, gan, color=True)

    return (X_train, y_train), (X_test, y_test)


# Function to load and pre-process CIFAR100 dataset
def get_cifar100(range=(0,1)):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = preprocess_data(X_train, range)
    X_test = preprocess_data(X_test, range)

    return (X_train, y_train), (X_test, y_test)