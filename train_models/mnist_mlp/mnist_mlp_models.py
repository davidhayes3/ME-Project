from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.regularizers import l1
from keras import backend as K
import numpy as np



# Define constants
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


def encoder_model():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(latent_dim))

    return model


def sparse_encoder_model():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512, activity_regularizer=l1(10e-5)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, activity_regularizer=l1(10e-5)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(latent_dim))

    return model


def vae_encoder_model():
    x = Input(shape=img_shape)

    x_enc = Flatten()(x)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)

    z_mean = Dense(latent_dim)(x_enc)
    z_log_var = Dense(latent_dim)(x_enc)

    return Model(x, [z_mean, z_log_var])


def generator_model(gan=False):
    model = Sequential()

    model.add(Dense(512, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape)))
    if gan is False:
        model.add(Activation('sigmoid'))
    if gan is not False:
        model.add(Activation('tanh'))
    model.add(Reshape(img_shape))

    return model


def context_generator_model(missing_shape):
    model = Sequential()

    model.add(Dense(512, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(missing_shape)))
    model.add(Activation('tanh'))
    model.add(Reshape(missing_shape))

    return model


def bigan_discriminator_model():

    z = Input(shape=(latent_dim,))
    img = Input(shape=img_shape)
    d_in = concatenate([z, Flatten()(img)])

    model = Dense(1024)(d_in)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation='sigmoid')(model)

    return Model([z, img], validity)


def gan_discriminator_model():
    img = Input(shape=img_shape)

    model = Flatten()(img)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation='sigmoid')(model)

    return Model(img, validity)


def aae_discriminator_model():
    z = Input(shape=(latent_dim,))

    model = Dense(1024)(z)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation='sigmoid')(model)

    return Model(z, validity)


def context_discriminator_model(missing_shape):
    img = Input(shape=missing_shape)

    model = Flatten()(img)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation='sigmoid')(model)

    return Model(img, validity)