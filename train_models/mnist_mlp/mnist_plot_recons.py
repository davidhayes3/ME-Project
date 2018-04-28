import numpy as np
import matplotlib.pyplot as plt
from mnist_mlp_models import encoder_model, generator_model
from functions.data_funcs import get_mnist
import matplotlib.gridspec as gridspec

# =====================================
# Define constants
# =====================================

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
num_classes = 10
image_path = 'Images/mnist_lr'
model_path = 'Models/mnist_lr'


# =====================================
# Load dataset
# =====================================

# Load MNIST data in range [-1,1]
(X_train, _), (X_test, y_test) = get_mnist(gan=True)


# Instantiate models
generator = generator_model(gan=True)
generator.load_weights('Models/mnist_bigan_generator.h5')
encoder = encoder_model()
encoder.load_weights('Models/mnist_bigan_encoder.h5')


# Get initial data examples to train on
classes = np.arange(num_classes)
test_digit_indices = np.empty(0)

# Modify training set to contain set number of labels for each class
for class_index in range(num_classes):
    # Generate training set with even class distribution over all labels
    indices = [i for i, y in enumerate(y_test) if y == classes[class_index]]
    indices = np.asarray(indices)
    indices = indices[0:10]
    test_digit_indices = np.concatenate((test_digit_indices, indices))

test_digit_indices = test_digit_indices.astype(np.int)

# Generate test and reconstructed digit arrays
X_test = X_test[test_digit_indices]

num_rows = 10
num_cols = 10

plt.figure(figsize=(num_rows, num_cols))

gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=[1,1,1,1,1,1,1,1,1,1],
         wspace=0., hspace=0., top=0.8, bottom=0.2, left=0.2, right=0.8)

for i in range(num_rows):
    for j in range(num_cols):
        im = X_test[i*num_rows + j].reshape(28,28)
        ax = plt.subplot(gs[i,j])
        plt.imshow(im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.savefig('mnist_test_digits')
