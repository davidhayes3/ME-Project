import numpy as np
import matplotlib.pyplot as plt
from cifar10_models import deterministic_encoder_model, generator_model, vae_encoder_model
from common_models.common_models import vae_encoder_sampling_model
from functions.data_funcs import get_cifar10
import matplotlib.gridspec as gridspec

# =====================================
# Define constants
# =====================================

img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 64
num_classes = 10
num_recons_per_class = 10


# =====================================
# Load dataset
# =====================================

# Load CIFAR-10 data in range [-1,1]
(X_train, _), (X_test, y_test) = get_cifar10()

# Get initial data examples to train on
classes = np.arange(num_classes)
test_digit_indices = np.empty(0)


# =====================================
# Choose examples from test set
# =====================================

#  test set to contain set number of labels for each class
for class_index in range(num_classes):
    # Generate training set with even class distribution over all labels
    indices = [i for i, y in enumerate(y_test) if y == classes[class_index]]
    indices = np.asarray(indices)
    indices = indices[0:num_recons_per_class]
    test_digit_indices = np.concatenate((test_digit_indices, indices))

test_digit_indices = test_digit_indices.astype(np.int)

# Generate test and reconstructed digit arrays
X_test = X_test[test_digit_indices]


# =====================================
# Plot test examples
# =====================================

num_rows = num_recons_per_class
num_cols = num_classes

plt.figure(figsize=(num_rows, num_cols))

gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=[1,1,1,1,1,1,1,1,1,1],
         wspace=0., hspace=0., top=0.8, bottom=0.2, left=0.2, right=0.8)

for i in range(num_rows):
    for j in range(num_cols):
        im = X_test[i*num_rows + j].reshape(img_shape)
        ax = plt.subplot(gs[i,j])
        plt.imshow(im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.savefig('cifar10_test_examples')


# =====================================
# Plot model reconstructions
# =====================================

generator = generator_model()

for model in ('basic_ae', 'dae', 'aae', 'bigan_determ',  'vae'):

    if model == 'vae':
        vae_encoder = vae_encoder_model()
        encoder = vae_encoder_sampling_model(vae_encoder, latent_dim, img_shape, epsilon_std=0.05)

    else:
        encoder = deterministic_encoder_model()

    encoder.load_weights('Models/cifar10_' + model + '_encoder.h5')
    generator.load_weights('Models/cifar10_' + model + '_generator.h5')

    recon_x = generator.predict(encoder.predict(X_test))

    num_rows = num_classes
    num_cols = num_recons_per_class

    plt.figure(figsize=(num_rows, num_cols))

    gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=num_recons_per_class*[1],
                           wspace=0., hspace=0., top=0.8, bottom=0.2, left=0.2, right=0.8)

    for i in range(num_rows):
        for j in range(num_cols):
            im = recon_x[i * num_rows + j].reshape(img_rows, img_cols, channels)
            ax = plt.subplot(gs[i, j])
            plt.imshow(im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig('Images/cifar10_' + model + '_recons.png')
