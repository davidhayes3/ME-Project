import numpy as np
from random import randint
from keras.datasets import mnist
from mnist_mlp_models import encoder_model, generator_model, vae_encoder_model
from common_models.common_models import vae_encoder_sampling_model
import matplotlib.pyplot as plt
from functions.data_funcs import get_mnist


# =====================================
# Define constants
# =====================================

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
num_classes = 10
num_steps = 7


# =====================================
# Load data
# =====================================

# Load dataset
(_, _), (x_test_gan, y_test) = get_mnist(gan=True)
(_, _), (x_test_ae, _) = get_mnist()


# =====================================
# Interpolate
# =====================================

model_names = ('basic_ae', 'dae', 'sae', 'vae', 'aae', 'lr', 'jlr', 'bigan', 'posthoc_bigan')

for i, model_name in enumerate(model_names):

    gan = False
    if i > 4:
        gan = True

    if gan is True:
        x_test = x_test_gan
    else:
        x_test = x_test_ae

    image_path = 'Images/mnist_' + model_name + '_ls_interpolations'
    encoder_path = 'Models/mnist_' + model_name + '_encoder.h5'
    generator_path = 'Models/mnist_' + model_name + '_generator.h5'

    if model_name == 'vae':
        vae_encoder = vae_encoder_model()
        encoder = vae_encoder_sampling_model(vae_encoder, latent_dim, img_shape, epsilon_std=0.05)
    else:
        encoder = encoder_model()

    encoder.load_weights(encoder_path)

    generator = generator_model(gan=gan)
    if model_name == 'lr':
        generator.load_weights('Models/mnist_gan_generator.h5')
    elif model_name == 'posthoc_bigan':
        generator.load_weights('Models/mnist_gan_generator.h5')
    else:
        generator.load_weights(generator_path)

    # Get sets of just 1 and 9 digits
    x_test_7 = x_test[y_test == 7]
    x_test_5 = x_test[y_test == 5]

    # Create micro batch
    X = np.array([x_test_7[8], x_test_5[7]])

    # Compute latent space projection
    latent_x = encoder.predict(X)
    latent_start, latent_end = latent_x

    # Get original image for comparison
    start_image, end_image = X

    vectors = []
    normal_images = []

    # Linear interpolation
    alpha_values = np.linspace(0, 1, num_steps)

    for alpha in alpha_values:

        # Latent space interpolation
        vector = latent_start * (1 - alpha) + latent_end * alpha
        vectors.append(vector)

        # Image space interpolation
        blend_image =  (1 - alpha) * start_image + alpha * end_image
        normal_images.append(blend_image)


    # Decode latent space vectors
    vectors = np.array(vectors)
    reconstructions = generator.predict(vectors)

    if gan is True:
        reconstructions = 0.5 * reconstructions + 0.5
    reconstructions *= 255


    # Convert pixel-space images for use in plotting
    normal_images = np.array(normal_images)


    # Plot interpolations
    plt.figure()
    n = len(reconstructions)

    for i in range(n):
        # Display interpolation in pixel-space
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(normal_images[i].reshape(img_rows, img_cols))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display interpolation in latent space
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].reshape(img_rows, img_cols))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(image_path)