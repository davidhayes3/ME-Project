from __future__ import print_function, division

from functions.data_funcs import get_mnist
from functions.auxiliary_funcs import save_models
from functions.visualization_funcs import plot_gan_epoch_loss, plot_gan_batch_loss, plot_discriminator_acc
from mnist_mlp_models import encoder_model, context_generator_model, context_discriminator_model
from common_models.common_models import autoencoder_model
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_rows = 28
img_cols = 28
mask_height = 8
mask_width = 8
channels = 1
img_shape = (img_rows, img_cols, channels)
missing_shape = (mask_height, mask_width, channels)
num_classes = 10
image_path = 'Images/mnist_ce'
model_path = 'Models/mnist_ce'

# =====================================
# Load dataset
# =====================================

# Load MNIST dataset in range [-1,1]
(X_train, y_train), (X_test, y_test) = get_mnist(gan=True)


# =====================================
# Define necessary functions
# =====================================

def sample_images(path, epoch, imgs):
    r, c = 3, 6

    masked_imgs, missing_parts, (y1, y2, x1, x2) = mask_randomly(imgs)
    gen_missing = generator.predict(encoder.predict(masked_imgs))

    imgs = 0.5 * imgs + 0.5
    masked_imgs = 0.5 * masked_imgs + 0.5
    gen_missing = 0.5 * gen_missing + 0.5

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0, i].imshow(imgs[i].reshape(img_rows, img_cols))
        axs[0, i].axis('off')
        axs[1, i].imshow(masked_imgs[i].reshape(img_rows, img_cols))
        axs[1, i].axis('off')
        filled_in = imgs[i].copy()
        filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
        axs[2, i].imshow(filled_in.reshape(img_rows, img_cols))
        axs[2, i].axis('off')
        plt.gray()
    fig.savefig(path + '_%d.png' % epoch)
    plt.close()


# Function to mask a random square of pixels in image
def mask_randomly(imgs):

    # Randomly choose co-ordinates for the masking of each image in imgs
    y1 = np.random.randint(0, img_rows - mask_height, imgs.shape[0])
    y2 = y1 + mask_height
    x1 = np.random.randint(0, img_rows - mask_width, imgs.shape[0])
    x2 = x1 + mask_width

    # Empty matrix for masked images
    masked_imgs = np.empty_like(imgs)
    # Empty array for masks
    missing_parts = np.empty((imgs.shape[0], mask_height, mask_width, channels))

    # Loop through all images
    for i, img in enumerate(imgs):
        # Copy full image to masked image
        masked_img = img.copy()
        # Determine co-ordinates to be masked for this particular image
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
        # Save mask in separate array
        missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
        # Remove mask from full image
        masked_img[_y1:_y2, _x1:_x2, :] = 0
        # Save masked image
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)


# =====================================
# Instantiate & compile models
# =====================================

# Instantiate models
encoder = encoder_model()
generator = context_generator_model(missing_shape)
context_generator = autoencoder_model(encoder, generator)
discriminator = context_discriminator_model(missing_shape)

# Specify optimizer for models
lr = 0.0002
beta_1 = 0.5
optimizer = Adam(lr=lr, beta_1=beta_1)

# Compile models
context_generator.compile(loss=['binary_crossentropy'], optimizer=optimizer)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# Define context encoder model
masked_img = Input(shape=img_shape)

enc_img = encoder(masked_img)
gen_mask = generator(enc_img)

validity = discriminator(gen_mask)

context_encoder = Model(masked_img, [gen_mask, validity])

# Compile model
discriminator.trainable = False
context_encoder.compile(loss=['mse', 'binary_crossentropy'],optimizer=optimizer)


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 50
batch_size = 128
epoch_save_interval = 5
num_batches = int(X_train.shape[0] / batch_size)

# Define arrays to hold progression of discriminator and bigan losses
d_batch_loss_trajectory = np.zeros(epochs * num_batches)
g_batch_loss_trajectory = np.zeros(epochs * num_batches)
d_epoch_loss_trajectory = np.zeros(epochs)
g_epoch_loss_trajectory = np.zeros(epochs)
d_acc_trajectory = np.zeros(epochs)


# Train for set number of epochs
for epoch in range(epochs):

    # Print current epoch number
    print("\nEpoch: " + str(epoch + 1) + "/" + str(epochs))

    # Set epoch losses to zero
    d_epoch_loss_sum = 0
    g_epoch_loss_sum = 0
    d_acc = 0

    # Shuffle training set
    new_permutation = np.random.randint(0, X_train.shape[0], X_train.shape[0])
    X_train = X_train[new_permutation]

    # Train on all batches
    for batch in range(num_batches):

        # Labels for supervised training
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select next batch of images from training set and encode
        imgs = X_train[batch * batch_size: (batch + 1) * batch_size]

        masked_imgs, missing_piece, _ = mask_randomly(imgs)

        # Generate a half batch of new images
        gen_missing_piece = generator.predict(encoder.predict(masked_imgs))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(missing_piece, valid)
        d_loss_fake = discriminator.train_on_batch(gen_missing_piece, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]
        d_acc += d_loss[1]

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        g_loss = context_encoder.train_on_batch(masked_imgs, [missing_piece, valid])

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch+1, batch, num_batches,
                                                                                    d_loss[0], 100 * d_loss[1],
                                                                                    g_loss[0]))


    # Record epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches
    d_acc_trajectory[epoch] = 100 * (d_acc / num_batches)

    # If at save interval, save generated image samples
    if epoch % epoch_save_interval == 0:
        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], 6)
        imgs = X_train[idx]
        sample_images(image_path, epoch, imgs)


# Save encoder weights
save_models(path=model_path, encoder=encoder)


# =====================================
# Visualizations
# =====================================

# Save loss curves
plot_gan_batch_loss(image_path, epochs, num_batches, d_batch_loss_trajectory, g_batch_loss_trajectory)
plot_gan_epoch_loss(image_path, epochs, d_epoch_loss_trajectory, g_epoch_loss_trajectory)
plot_discriminator_acc(image_path, epochs, d_acc_trajectory)