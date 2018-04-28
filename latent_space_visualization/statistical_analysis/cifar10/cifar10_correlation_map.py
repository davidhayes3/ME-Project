
import seaborn
from keras.datasets import cifar10
import numpy as np
import keras.utils
import matplotlib.pyplot as plt
from cifar10_models import deterministic_encoder_model

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

# Get correlation coefficients of all latent dimensions for entire training set
one_latent_dim_interclass_correlations = np.corrcoef([set[:,0] for set in latent_sets])
training_set_latent_correlations = np.corrcoef([latent_spaces[:,i] for i in range(latent_dim)])

# Remove duplicate correlation from array, through use of mask
mask = np.zeros_like(one_latent_dim_interclass_correlations)
mask[np.triu_indices_from(mask)] = True
mask[np.diag_indices_from(mask)] = False

values = np.arange(0.5, num_classes+0.5, 1)
names = ['Plane','AM','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for i in range(latent_dim):
    plt.figure()

    one_latent_dim_interclass_correlations = np.corrcoef([set[:,i] for set in latent_sets])
    seaborn.heatmap(one_latent_dim_interclass_correlations, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)
    plt.yticks(values, names, rotation=0)
    plt.xticks(values, names, rotation=90)

    plt.savefig('Images/cifar10_interclass_corr_latent_%d' % i)
    plt.close()


# Remove duplicate correlation from array, through use of mask
mask = np.zeros_like(training_set_latent_correlations)
mask[np.triu_indices_from(mask)] = True
mask[np.diag_indices_from(mask)] = False


# Create heatmap of correlation coefficients
plt.figure()
seaborn.heatmap(training_set_latent_correlations, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)

# Change orientation of labels for easier readability
plt.yticks(rotation=0)
plt.xticks(rotation=90)

# Label Axes
plt.xlabel('Latent Dimension')
plt.ylabel('Latent Dimension')

# Save plot
plt.savefig('cifar10_training_set_latent_corrs')


# Plot histogram of correlation distribution
plt.figure()

plt.hist(training_set_latent_correlations, 100, facecolor='green', alpha=0.5)
plt.xlim(-0.4, 0.4)
#plt.ylim(0, 500)


plt.savefig('cifra10_training_set_corrs_distrib')