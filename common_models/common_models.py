from keras.layers import Input, Lambda
from keras.models import Sequential, Model
from keras import backend as K


def bigan_model(generator, encoder, discriminator, latent_dim, img_shape):
    z = Input(shape=(latent_dim,))
    x = Input(shape=img_shape)

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


def aae_model(encoder, decoder, discriminator, img_shape):
    x = Input(shape=img_shape)

    enc_x = encoder(x)
    recon_x = decoder(enc_x)

    validity = discriminator(enc_x)

    return Model(x, [recon_x, validity])


def latent_reconstructor_model(d, e):
    model = Sequential()

    model.add(d)
    model.add(e)

    return model


def vae_encoder_sampling_model(encoder, latent_dim, img_shape, epsilon_std):
    x = Input(shape=img_shape)

    # Define sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    z_mean, z_log_var = encoder(x)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    return Model(x, z)


def vae_model(vae_encoder_sample, generator, img_shape):
    x = Input(shape=img_shape)

    z = vae_encoder_sample(x)

    recon_x = generator(z)

    return Model(x, recon_x)