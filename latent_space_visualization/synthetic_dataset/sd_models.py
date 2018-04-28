from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, LeakyReLU, Dropout, concatenate, BatchNormalization, Lambda, Flatten, Reshape
from keras.regularizers import l1
import keras.backend as K
import numpy as np

# =====================================
# Define constants
# =====================================

img_rows = 2
img_cols = 2
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 2


# =====================================
# Define models
# =====================================

def encoder_model():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
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
    x_enc = BatchNormalization(momentum=0.8)(x_enc)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)
    x_enc = BatchNormalization(momentum=0.8)(x_enc)


    z_mean = Dense(latent_dim)(x_enc)
    z_log_var = Dense(latent_dim)(x_enc)

    return Model(x, [z_mean, z_log_var])


def generator_model(gan=False):
    model = Sequential()

    model.add(Dense(512, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape)))
    model.add(Reshape(img_shape))
    if gan is False:
        model.add(Activation('sigmoid'))
    if gan is not False:
        model.add(Activation('tanh'))

    return model


def bigan_discriminator_model():
    z_in = Input(shape=(latent_dim,))
    z = Dense(512)(z_in)
    z = LeakyReLU(alpha=0.2)(z)
    z = Dropout(0.5)(z)
    z = Dense(512)(z)
    z = LeakyReLU(alpha=0.2)(z)

    x_in = Input(shape=img_shape)
    x = Flatten()(x_in)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)

    c = concatenate([z, x])
    c = Dropout(0.5)(c)
    c = Dense(1024)(c)
    c = LeakyReLU(alpha=0.2)(c)
    c = Dropout(0.5)(c)
    c = Dense(1)(c)
    validity = Activation('sigmoid')(c)

    return Model([z_in, x_in], validity)


def gan_discriminator_model():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def aae_discriminator_model():
    model = Sequential()

    model.add(Dense(512, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model



def bigan_model(generator, encoder, discriminator):
    z = Input(shape=(latent_dim,))
    x = Input(shape=(np.prod(img_shape),))

    x_ = generator(z)
    z_ = encoder(x)

    fake = discriminator([z, x_])
    valid = discriminator([z_, x])

    return Model([z, x], [fake, valid])


def gan_model(generator, discriminator):
    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    return model


def autoencoder_model(encoder, decoder):
    model = Sequential()

    model.add(encoder)
    model.add(decoder)

    return model


def aae_model(encoder, decoder, discriminator):
    x = Input(shape=(np.prod(img_shape),))

    enc_x = encoder(x)
    recon_x = decoder(enc_x)

    validity = discriminator(enc_x)

    return Model(x, [recon_x, validity])


def latent_reconstructor_model(d, e):
    model = Sequential()

    model.add(d)
    model.add(e)

    return model


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.0)

    return z_mean + K.exp(z_log_var / 2) * epsilon


def vae_model(encoder, generator):
    x = Input(shape=img_shape)

    z_mean, z_log_var = encoder(x)

    z = Lambda(sampling)([z_mean, z_log_var])

    recon_x = generator(z)

    return Model(x, recon_x)
