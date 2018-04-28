from __future__ import print_function, division
import numpy as np
from keras.optimizers import Adam
from functions.data_funcs import get_cifar10
from cifar10_models import encoder_model, generator_model, bigan_discriminator_model
from common_models.common_models import bigan_model
from functions.auxiliary_funcs import save_models
from functions.visualization_funcs import plot_gan_epoch_loss, plot_gan_batch_loss, \
    plot_discriminator_acc, save_reconstructions, save_imgs


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 64
num_classes = 10
image_path = 'Images/cifar10_posthoc_bigan'
model_path = 'Models/cifar10_posthoc_bigan'


# =====================================
# Load dataset
# =====================================

(X_train, _), (X_test, y_test) = get_cifar10()


# =====================================
# Instantiate and compile models
# =====================================

generator = generator_model()
generator.load_weights('Models/cifar10_gan_generator.h5')
encoder = encoder_model()
discriminator = bigan_discriminator_model()


lr = 0.0002
beta_1 = 0.5
opt_d = Adam(lr=lr, beta_1=beta_1)
opt_g = Adam(lr=lr, beta_1=beta_1)


generator.trainable = False
encoder.trainable = False
bigan_discriminator = bigan_model(generator, encoder, discriminator)
bigan_discriminator.compile(optimizer=opt_d, loss='binary_crossentropy')


encoder.trainable = True
discriminator.trainable = False
bigan_generator = bigan_model(generator, encoder, discriminator)
bigan_generator.compile(optimizer=opt_g, loss='binary_crossentropy')


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 1000
batch_size = 128
epoch_save_interval = 5

# Compute number of batches in one epoch
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

        # Select next batch of images from training set
        imgs = X_train[batch * batch_size: (batch + 1) * batch_size]
        # Generator normal distributed latent vector
        z = np.random.normal(size=(batch_size, latent_dim))

        # Create labels for discriminator inputs
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss = bigan_discriminator.train_on_batch([z, imgs], [fake, valid])

        # Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]
        d_acc += (100 * d_loss[1])

        # ----------------------------
        #  Train Generator and Encoder
        # ----------------------------

        # Train the generator (z -> img_ is valid and img -> z_ is is invalid)
        ge_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake])

        g_batch_loss_trajectory[epoch * num_batches + batch] = ge_loss[0]
        g_epoch_loss_sum += ge_loss[0]

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, batch, num_batches,
            d_loss[0], 100 * d_loss[1], ge_loss[0]))


    # Record epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches
    d_acc_trajectory = d_acc / num_batches

    # If at save interval, save generated image samples
    if epoch % epoch_save_interval == 0:
        z = np.random.normal(size=(25, latent_dim))
        gen_imgs = generator.predict(z)
        save_imgs(image_path, gen_imgs, epoch, img_rows, img_cols, channels, color=False)


# Save models to file
save_models(path=model_path, encoder=encoder)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=True)

# Save loss curves
plot_gan_batch_loss(image_path, epochs, num_batches, d_batch_loss_trajectory, g_batch_loss_trajectory)
plot_gan_epoch_loss(image_path, epochs, d_epoch_loss_trajectory, g_epoch_loss_trajectory)
plot_discriminator_acc(image_path, epochs, d_acc_trajectory)