from __future__ import print_function, division
from cifar10_models import deterministic_encoder_model, generator_model, aae_discriminator_model
from common_models.common_models import aae_model
from functions.visualization_funcs import save_reconstructions, plot_gan_batch_loss, plot_gan_epoch_loss, \
    plot_discriminator_acc, save_imgs
from functions.data_funcs import get_cifar10
from functions.auxiliary_funcs import save_models
from keras.optimizers import Adam
import numpy as np


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
image_path = 'Images/cifar10_aae'
model_path = 'Models/cifar10_aae'


# =====================================
# Load dataset
# =====================================

(X_train, _), (X_test, y_test) = get_cifar10()


# =====================================
# Instantiate and compile models
# =====================================

encoder = deterministic_encoder_model()
generator = generator_model()
discriminator = aae_discriminator_model()

lr = 0.0002
beta_1 = 0.5
optimizer = Adam(lr=lr, beta_1=beta_1)

# Compile discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Compile AAE
discriminator.trainable = False
aae = aae_model(encoder, generator, discriminator, img_shape)
aae.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[0.99, 0.01],
            optimizer=optimizer)


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 50
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

        imgs = X_train[batch * batch_size: (batch + 1) * batch_size]
        latent_fake = encoder.predict(imgs)

        latent_real = np.random.normal(size=(batch_size, latent_dim))

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        d_loss_real = discriminator.train_on_batch(latent_real, valid)
        d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]
        d_acc += d_loss[1]

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        g_loss = aae.train_on_batch(imgs, [imgs, valid])

        # Record generator loss
        g_batch_loss_trajectory[epoch * num_batches + batch] = g_loss[0]
        g_epoch_loss_sum += g_loss[0]

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch + 1, batch, num_batches,
                                                                                      d_loss[0], 100 * d_loss[1],
                                                                                      g_loss[0]))

    # Get epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches
    d_acc_trajectory[epoch] = 100 * (d_acc / num_batches)

    # If at save interval, save generated image samples
    if epoch % epoch_save_interval == 0:
        # Generate random sample of latent vectors and save generated images
        z = np.random.normal(size=(25, latent_dim))
        gen_imgs = generator.predict(z)
        save_imgs(image_path, gen_imgs, epoch, img_rows, img_cols, channels, color=True)
        # Save visualization of 2D latent space


# Save encoder weights
save_models(path=model_path, encoder=encoder, generator=generator)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=True)

# Save loss curves
plot_gan_batch_loss(image_path, epochs, num_batches, d_batch_loss_trajectory, g_batch_loss_trajectory)
plot_gan_epoch_loss(image_path, epochs, d_epoch_loss_trajectory, g_epoch_loss_trajectory)
plot_discriminator_acc(image_path, epochs, d_acc_trajectory)