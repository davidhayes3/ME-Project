from keras.layers import *
from keras.models import Sequential, Model


def encoder_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(128, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(256, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(512, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(512, (1,1), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, (1,1), strides=(1,1)))
    model.add(Activation('linear'))
    return model


def generator_model():
    model = Sequential()
    model.add(Conv2DTranspose(256, kernel_size=(4,4), strides=(1,1), input_shape=(1,1,64)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(64, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(32, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(32, (5,5), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32, (1, 1), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(3, (1, 1), strides=(1,1)))
    model.add(Activation('sigmoid'))
    return model


def denc_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), strides=(1,1), input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (4,4), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (4,4), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (4,4), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (4,4), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    return model


def dgen_model():
    model = Sequential()
    model.add(Conv2D(512, (1,1), strides=(1,1), input_shape=(1,1,64)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, (1,1), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Merge([denc_model(), dgen_model()], mode='concat'))
    model.add(Conv2D(1024, (1,1), strides=(1,1), input_shape=(1,1,1024)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(1024, (1,1), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (1,1), strides=(1,1)))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    return model


def generator_containing_discriminator(g,d):
    g_in = Input(shape=(1,1,64))
    g_out = g(g_in)
    d.trainable=False
    d_out = d([g_out, g_in])
    model = Model(inputs=g_in, outputs=d_out)
    return model


def encoder_containing_discriminator(e,d):
    e_in = Input(shape=(32,32,3))
    e_out = e(e_in)
    d.trainable=False
    d_out = d([e_in, e_out])
    model = Model(inputs=e_in, outputs=d_out)
    return model