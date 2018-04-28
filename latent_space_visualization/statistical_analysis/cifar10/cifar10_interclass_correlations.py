import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from keras.datasets import cifar10
import numpy as np
import keras.utils
import matplotlib.pyplot as plt
from cifar10_models import encoder_model, deterministic_encoder_model
from scipy.stats.stats import pearsonr

# Define constants
num_classes = 10
latent_dim = 64

# Load saved models for encoder and decoder

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

# Create empty array for correlations
correlations = np.zeros((latent_dim, latent_dim))

# Examine correlations between latent dimensions for two particular classes
for i in range(latent_dim):
    for j in range(latent_dim):
        correlations[i, j] = np.corrcoef(latent_cat[:, i], latent_dog[:, j])[0][1]
        #correlations[i,j] = np.corrcoef(latent_ship[:,i], latent_automobile[:,j])[0][1]

# Create heatmap of correlation coefficients
seaborn.heatmap(correlations, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, linewidths=2.5)

# Change orientation of labels for easier readability
plt.yticks(rotation=0)
plt.xticks(rotation=90)

# Label axes
plt.xlabel('Cat')
plt.ylabel('Dog')

# Save plots
plt.savefig('cifar10_cat_dog_latent')
#plt.savefig('cifar10_ship_automobile_latent')