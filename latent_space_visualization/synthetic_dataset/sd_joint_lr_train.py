from __future__ import print_function, division
import numpy as np
from keras.optimizers import Adam
from sd_models import encoder_model, generator_model, gan_discriminator_model
from common_models.common_models import latent_reconstructor_model, gan_model
from functions.auxiliary_funcs import save_models
from functions.visualization_funcs import plot_gan_epoch_loss, plot_gan_batch_loss, plot_discriminator_acc, \
    save_reconstructions, save_imgs, save_latent_vis


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
image_path = 'Images/sd_jlr'
model_path = 'Models/sd_jlr'


# =====================================
# Load dataset
# =====================================

# Load dataset
X_train = np.loadtxt('Dataset/synthetic_dataset_x_train.txt', dtype=np.float32)
X_test = np.loadtxt('Dataset/synthetic_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/synthetic_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/synthetic_dataset_y_test.txt', dtype=np.int)

# Reshape data to image format
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)

# Normalize data to (-1,1)
X_train = (X_train - 0.5) / 0.5
X_test = (X_test - 0.5) / 0.5



# =====================================
# Instantiate & compile models
# =====================================

# Instantiate models
generator = generator_model(gan=True)
encoder = encoder_model()
discriminator = gan_discriminator_model()
latent_reconstructor = latent_reconstructor_model(generator, encoder)

# Specify optimizer for models
lr = 0.0002
beta_1 = 0.5
optimizer = Adam(lr=lr, beta_1=beta_1)

# Compile discriminator
discriminator.compile(loss=['binary_crossentropy'],
                           optimizer=optimizer,
                           metrics=['accuracy'])

# Compile GAN
discriminator.trainable = False
gan_generator = gan_model(generator, discriminator)
gan_generator.compile(loss=['binary_crossentropy'],
                       optimizer=optimizer)

# Compile latent regressor
generator.trainable = False
latent_reconstructor.compile(optimizer='SGD', loss='mse')


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 100
batch_size = 128
epoch_save_interval = 5
num_batches = int(X_train.shape[0] / batch_size)

# Define arrays to hold progression of discriminator and bigan losses
d_batch_loss_trajectory = np.zeros(epochs * num_batches)
g_batch_loss_trajectory = np.zeros(epochs * num_batches)
lr_batch_loss_trajectory = np.zeros(epochs * num_batches)
d_epoch_loss_trajectory = np.zeros(epochs)
g_epoch_loss_trajectory = np.zeros(epochs)
lr_epoch_loss_trajectory = np.zeros(epochs)
d_acc_trajectory = np.zeros(epochs)


# Train for set number of epochs
for epoch in range(epochs):

    # Print current epoch number
    print("\nEpoch: " + str(epoch + 1) + "/" + str(epochs))

    # Set epoch losses to zero
    d_epoch_loss_sum = 0
    g_epoch_loss_sum = 0
    lr_epoch_loss_sum = 0
    d_acc = 0

    # Train on all batches
    for batch in range(num_batches):

        # Create labels for discriminator inputs
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Select next batch of images from training set and encode
        imgs = X_train[batch * batch_size: (batch + 1) * batch_size]

        # Sample noise and generate img
        z = np.random.normal(size=(batch_size, latent_dim))
        imgs_ = generator.predict(z)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(imgs_, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]
        d_acc += d_loss[1]

        # ----------------------------
        #  Train Generator
        # ----------------------------

        # Train the generator (z -> img is valid and img -> z is is invalid)
        g_loss = gan_generator.train_on_batch(z, valid)

        # Record generator batch loss details
        g_batch_loss_trajectory[epoch * num_batches + batch] = g_loss
        g_epoch_loss_sum += g_loss

        # ----------------------------
        #  Train Encoder
        # ----------------------------

        lr_loss = latent_reconstructor.train_on_batch(z, z)

        lr_batch_loss_trajectory[epoch * num_batches + batch] = lr_loss
        lr_epoch_loss_sum += lr_loss

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch + 1, batch, num_batches,
                                                                                      d_loss[0], 100 * d_loss[1],
                                                                                      g_loss))

    # Record epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches
    lr_epoch_loss_trajectory[epoch] = lr_epoch_loss_sum / num_batches
    d_acc_trajectory[epoch] = 100 * (d_acc / num_batches)


    # If at save interval, save generated image samples
    if epoch % epoch_save_interval == 0:
        # Generate images from prior sample
        z = np.random.normal(size=(25, latent_dim))
        gen_imgs = generator.predict(z)
        save_imgs(image_path, gen_imgs, epoch, img_rows, img_cols, channels, color=False)
        # Save latent visualization
        save_latent_vis(image_path, X_train, y_train, encoder, num_classes, epoch)


# Save learned generator model to file
save_models(path=model_path, generator=generator, encoder=encoder)


# =====================================
# Visualization
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=False)

# Save loss curves
plot_gan_batch_loss(image_path, epochs, num_batches, d_batch_loss_trajectory, g_batch_loss_trajectory)
plot_gan_epoch_loss(image_path, epochs, d_epoch_loss_trajectory, g_epoch_loss_trajectory)
plot_discriminator_acc(image_path, epochs, d_acc_trajectory)