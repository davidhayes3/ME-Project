import numpy as np
from random import randint
from keras.datasets import mnist
from mnist_conv_ae_models import *
import matplotlib.pyplot as plt
import scipy.stats


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

num_steps = 7

# Shows linear inteprolation in image space vs latent space
print("Generating interpolations...")

# Create micro batch
X = np.array([x_test[randint(0,x_test.shape[0])], x_test[randint(0,x_test.shape[0])]])

# Generate encoder and decoder models
encoder = encoder_model()
encoder.load_weights('encoder.h5')
decoder = decoder_model()
decoder.load_weights('decoder.h5')

# Compute latent space projection
latentX = encoder.predict(X)
latentStart, latentEnd = latentX

# Get original image for comparison
startImage, endImage = X

vectors = []
normalImages = []

# Linear interpolation
alphaValues = np.linspace(0, 1, num_steps)

for alpha in alphaValues:

    # Latent space interpolation
    vector = latentStart * (1 - alpha) + latentEnd * alpha
    vectors.append(vector)

    # Image space interpolation
    blendImage =  (1 - alpha) * startImage + alpha * endImage
    normalImages.append(blendImage)


# Decode latent space vectors
vectors = np.array(vectors)
reconstructions = decoder.predict(vectors)
reconstructions *= 255

# Convert pixel-space images for use in plotting
normalImages = np.array(normalImages)


# Plot interpolations
plt.figure(figsize=(20, 4))
n = len(reconstructions)

for i in range(n):
    # Display interpolation in pixel-space
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(normalImages[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display interpolation in latent space
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructions[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()