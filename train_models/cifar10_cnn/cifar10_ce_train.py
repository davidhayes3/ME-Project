from __future__ import print_function, division

from functions.data_funcs import get_mnist
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_rows = 32
img_cols = 32
mask_height = 8
mask_width = 8
channels = 3
img_shape = (img_rows, img_cols, channels)
missing_shape = (mask_height, mask_width, channels)
num_classes = 10
image_path = 'Images/cifar10_ce'
model_path = 'Models/cifar10_ce'



def sample_images(path, epoch, imgs):
    r, c = 3, 6

    masked_imgs, missing_parts, (y1, y2, x1, x2) = mask_randomly(imgs)
    gen_missing = generator.predict(masked_imgs)

    imgs = 0.5 * imgs + 0.5
    masked_imgs = 0.5 * masked_imgs + 0.5
    gen_missing = 0.5 * gen_missing + 0.5

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0, i].imshow(imgs[i, :, :])
        axs[0, i].axis('off')
        axs[1, i].imshow(masked_imgs[i, :, :])
        axs[1, i].axis('off')
        filled_in = imgs[i].copy()
        filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
        axs[2, i].imshow(filled_in)
        axs[2, i].axis('off')
    fig.savefig(path + '_%d.png' % epoch)
    plt.close()


def generator_model():
    model = Sequential()
    # Encoder
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(512, kernel_size=1, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    # Decoder
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation('tanh'))

    masked_img = Input(shape=img_shape)
    gen_missing = model(masked_img)

    return Model(masked_img, gen_missing)


def discriminator_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=missing_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=missing_shape)
    validity = model(img)

    return Model(img, validity)

    return model


def mask_randomly(imgs):
    y1 = np.random.randint(0, img_rows - mask_height, imgs.shape[0])
    y2 = y1 + mask_height
    x1 = np.random.randint(0, img_rows - mask_width, imgs.shape[0])
    x2 = x1 + mask_width

    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((imgs.shape[0], mask_height, mask_width, channels))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
        missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
        masked_img[_y1:_y2, _x1:_x2, :] = 0
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)


# =====================================
# Instantiate & compile models
# =====================================

# Instantiate models
generator = generator_model()
discriminator = discriminator_model()

# Specify optimizer for models
lr = 0.0002
beta_1 = 0.5
optimizer = Adam(lr=0.0002, beta_1=beta_1)

# Compile models
generator.compile(loss=['binary_crossentropy'], optimizer=optimizer)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define context encoder model
masked_img = Input(shape=img_shape)

gen_mask = generator(masked_img)

validity = discriminator(gen_mask)

# The combined model (stacked generator and discriminator) takes
# masked_img as input => generates missing image => determines validity
context_encoder = Model(masked_img, [gen_mask, validity])

discriminator.trainable = False
context_encoder.compile(loss=['mse', 'binary_crossentropy'],optimizer=optimizer)


# =====================================
# Load dataset
# =====================================

# Load CIFAR10 dataset
(X_train, y_train), (X_test, y_test) = get_mnist(gan=True)


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
    perm = np.random.randint(0, X_train.shape[0], X_train.shape[0])
    X_train = X_train[perm]

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
        gen_missing_piece = generator.predict(masked_imgs)

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
            d_loss[0], 100 * d_loss[1], g_loss[0]))


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