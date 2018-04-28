import os
import sys
import h5py
# import cv2
import math
import random, string

from matplotlib.pyplot import cm
import numpy as np
from scipy.stats import norm
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D

from mnist_conv_ae_models import encoder_model




def loadDataset():
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape([-1, 28, 28, 1]) / 255.
    X_test = X_test.reshape([-1, 28, 28, 1]) / 255.

    return (X_train, y_train), (X_test, y_test)


def plotEmbeddings3D(embeddings, y_sample, labels, num_classes):
    print('Plotting in 3D...')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.Spectral(np.linspace(0, 1, num_classes))

    xx = embeddings[:, 0]
    yy = embeddings[:, 1]
    zz = embeddings[:, 2]

    # plot the 3D data points
    for i in range(num_classes):
        ax.scatter(xx[y_sample == i], yy[y_sample == i], zz[y_sample == i], color=colors[i], label=labels[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.show()


def plotEmbeddings2D(embeddings, y_sample, labels, num_classes, with_images=False):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))

    xx = embeddings[:, 0]
    yy = embeddings[:, 1]

    # plot the images
    if with_images == True:
        for i, (x, y) in enumerate(zip(xx, yy)):
            im = OffsetImage(X_sample[i], zoom=0.1, cmap='gray')
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(np.column_stack([xx, yy]))
        ax.autoscale()

    # plot the 2D data points
    for i in range(num_classes):
        ax.scatter(xx[y_sample==i], yy[y_sample==i], color=colors[i], label=labels[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.show()



# Show dataset images with T-sne projection of latent space encoding
def computeLatentSpaceEmbeddings(X, encoder, num_dimensions):

    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder.predict(X)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=num_dimensions, init='pca')#, random_state=0)
    embeddings = tsne.fit_transform(X_encoded)

    return embeddings


# Show dataset images with T-sne projection of pixel space
def computePixelSpaceEmbeddings(X, num_dimensions):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=num_dimensions, init='pca')#, random_state=0)
    embeddings = tsne.fit_transform(X.reshape([-1, imageSize * imageSize * 1]))

    return embeddings


## Run visualizations

imageSize = 28
latent_dim = 32
num_dimensions = 3
num_classes = 10
num_samples = 10000

labels = np.arange(num_classes)

# Load dataset to test
print("Loading dataset...")
(X_train, y_train), (X_test, y_test) = loadDataset()

X_sample = X_test[:num_samples]
y_sample = y_test[:num_samples]

print(X_test.shape)
print(X_sample.shape)

encoder = encoder_model()
encoder.load_weights('mnist_conv_ae_encoder.h5')

latent_embeddings = computeLatentSpaceEmbeddings(X_sample, encoder, num_dimensions)
pixel_embeddings = computePixelSpaceEmbeddings(X_sample, num_dimensions)

#plotEmbeddings3D(latent_embeddings, y_sample, labels, num_classes)
#plotEmbeddings3D(pixel_embeddings, y_sample, labels, num_classes)

plotEmbeddings2D(latent_embeddings, y_sample, labels, num_classes)
plotEmbeddings2D(pixel_embeddings, y_sample, labels, num_classes)
