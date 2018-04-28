from __future__ import print_function, division
import numpy as np
from keras.optimizers import Adam
from sd_models import encoder_model, generator_model, bigan_discriminator_model
from common_models.common_models import bigan_model
from functions.auxiliary_funcs import save_models
from functions.visualization_funcs import plot_gan_epoch_loss, plot_gan_batch_loss, \
    plot_discriminator_acc, save_reconstructions, save_imgs, save_latent_vis


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_dim = 4
img_rows = 2
img_cols = 2
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 2
num_classes = 16
image_path = 'Images/sd_bigan'
model_path = 'Models/sd_bigan'


# =====================================
# Load dataset
# =====================================

# Load dataset
X_train = np.loadtxt('Dataset/synthetic_dataset_x_train.txt', dtype=np.float32)
X_test = np.loadtxt('Dataset/synthetic_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/synthetic_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/synthetic_dataset_y_test.txt', dtype=np.int)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
X_train_original = X_train

# Normalize data to (-1,1)
X_train = (X_train - 0.5) / 0.5
X_test = (X_test - 0.5) / 0.5


# =====================================
# Instantiate and compile models
# =====================================

# Instantiate models
generator = generator_model(gan=True)
encoder = encoder_model()
discriminator = bigan_discriminator_model()

# Specify optimizers for models
lr = 0.0002
beta_1 = 0.5
opt_d = Adam(lr=lr, beta_1=beta_1)
opt_g = Adam(lr=lr, beta_1=beta_1)

# Only discriminator is trainable when training discriminator
generator.trainable = False
encoder.trainable = False
bigan_discriminator = bigan_model(generator, encoder, discriminator, latent_dim, img_shape)
bigan_discriminator.compile(optimizer=opt_d, loss='binary_crossentropy')

# Discriminator is frozen when training generator and encoder
generator.trainable = True
encoder.trainable = True
discriminator.trainable = False
bigan_generator = bigan_model(generator,encoder, discriminator, latent_dim, img_shape)
bigan_generator.compile(optimizer=opt_g, loss='binary_crossentropy')


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 100
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
        valid = np.ones(batch_size)
        fake = np.zeros(batch_size)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss = bigan_discriminator.train_on_batch([z, imgs], [fake, valid])

        # Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]
        d_acc += d_loss[1]

        # ----------------------------
        #  Train Generator and Encoder
        # ----------------------------

        # Train the generator (z -> img_ is valid and img -> z_ is is invalid)
        ge_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake])

        g_batch_loss_trajectory[epoch * num_batches + batch] = ge_loss[0]
        g_epoch_loss_sum += ge_loss[0]

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, batch, num_batches,
                                                                                      d_loss[0], 50 * d_loss[1],
                                                                                      ge_loss[0]))


    # Record epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches
    d_acc_trajectory[epoch] = 100 * (d_acc / num_batches)

    # If at save interval, save generated image samples
    if epoch % epoch_save_interval == 0:
        # Generate random sample of latent vectors and save generated images
        z = np.random.normal(size=(25, latent_dim))
        gen_imgs = generator.predict(z)
        save_imgs(image_path, gen_imgs, epoch, img_rows, img_cols, channels, color=False)
        # Save visualization of 2D latent space
        save_latent_vis(image_path, X_train_original, y_train, encoder, num_classes, epoch)


# Save models to file
save_models(path=model_path, encoder=encoder, generator=generator)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=False)

# Save loss curves
plot_gan_batch_loss(image_path, epochs, num_batches, d_batch_loss_trajectory, g_batch_loss_trajectory)
plot_gan_epoch_loss(image_path, epochs, d_epoch_loss_trajectory, g_epoch_loss_trajectory)
plot_discriminator_acc(image_path, epochs, d_acc_trajectory)