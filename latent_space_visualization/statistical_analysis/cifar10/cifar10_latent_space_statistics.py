from keras.datasets import cifar10
import numpy as np
import keras.utils
import matplotlib.pyplot as plt
from cifar10_models import encoder_model, deterministic_encoder_model


# Define constants
num_classes = 10
latent_dim = 64

encoder = deterministic_encoder_model()
encoder.load_weights('cifar10_bigan_determ_encoder.h5')


# Load MNIST data and split into train and test set
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)
y_train = y_train.reshape((y_train.shape[0]))

# Encoder training set
latent_spaces = encoder.predict(X_train)

# Get max and min value of entire set for later plotting purposes
max = np.max(latent_spaces)
min = np.min(latent_spaces)

# Split training set into classes
latent_plane = latent_spaces[y_train == 0]
latent_automobile = latent_spaces[y_train == 1]
latent_bird = latent_spaces[y_train == 2]
latent_cat = latent_spaces[y_train == 3]
latent_deer = latent_spaces[y_train == 4]
latent_dog = latent_spaces[y_train == 5]
latent_frog = latent_spaces[y_train == 6]
latent_horse = latent_spaces[y_train == 7]
latent_ship = latent_spaces[y_train == 8]
latent_truck = latent_spaces[y_train == 9]


# Create list of all latent arrays
latent_sets = (latent_plane, latent_automobile, latent_bird, latent_cat, latent_deer, latent_dog, latent_frog,
               latent_horse, latent_ship, latent_truck)


plt.figure()

for i in range(latent_dim):
    ax = plt.subplot(8, 8, i + 1)
    plt.hist(latent_spaces[:,i], 100, facecolor='green', alpha=0.5)
    plt.xlim(min, max)
    plt.ylim(0, 2000)
    if i != 56:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.savefig('Images/cifar10_bigan_encoder_latent_distribution_training_set')


# Generate distribution of each latent dimension for each individual class
for i, set in enumerate(latent_sets):
    plt.figure()

    for j in range(latent_dim):
        ax = plt.subplot(8, 8, j + 1)
        plt.hist(set[:,j], 100, facecolor='green', alpha=0.5)
        plt.xlim(min, max)
        plt.ylim(0, 200)
        if j != 56:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig('Images/cifar10_bigan_encoder_latent_distribution_class_%d' % i)