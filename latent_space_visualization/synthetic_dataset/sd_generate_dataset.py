'''Script used to generate a synthetic dataset to enable 2D latent space visualizations'''

from __future__ import print_function
import numpy as np

# Settings
latent_dim = 2
img_rows = 2
img_cols = 2
channels = 1
img_shape = (img_rows, img_cols, channels)
num_classes = 16
num_train_examples = 5000 * num_classes
num_test_examples = 1000 * num_classes
variance = 0.07

# Create label arrays
y_train = np.random.choice(list(range(num_classes)), size=(num_train_examples,))
y_test = np.random.choice(list(range(num_classes)), size=(num_test_examples,))

# Create zero arrays for data
X_train = np.zeros((num_train_examples, np.prod(img_shape)))
X_test = np.zeros((num_test_examples, np.prod(img_shape)))

# Create data as binary version of label e.g. label 9 -> 1001
for i, y in enumerate(y_train):
    X_train[i] = np.array([int(x) for x in list('{:04b}'.format(y))])
for i, y in enumerate(y_test):
    X_test[i] = np.array([int(x) for x in list('{:04b}'.format(y))])

# Corrupt data with noise to add distinguish between samples and clip images to retain pixel values between 0 and 1
noise_factor = 0.07
X_train = X_train + noise_factor * np.random.normal(0., 1, size=X_train.shape)
X_test = X_test + noise_factor * np.random.normal(0., 1, size=X_test.shape)
X_train = np.clip(X_train, 0., 1.)
X_test = np.clip(X_test, 0., 1.)

# Save dataset
for data, name in [(X_train, 'x_train'), (X_test, 'x_test')]:
    np.savetxt('Dataset/synthetic_dataset_' + name + '.txt', data, fmt='%f')

for data, name in [(y_train, 'y_train'), (y_test, 'y_test')]:
    np.savetxt('Dataset/synthetic_dataset_' + name + '.txt', data, fmt='%d')
