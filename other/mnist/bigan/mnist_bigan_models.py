from keras.layers import *
from keras.models import Sequential, Model


# Define generator model
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

# Define encoder model
def encoder_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(100, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(100, (7, 7)))
    model.add(Activation('tanh'))
    model.add(Flatten())
    return model

def dgen_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 5 * 5))
    model.add(Activation('tanh'))
    model.add(Reshape((5, 5, 128), input_shape=(128 * 5 * 5,)))
    model.add(Activation('tanh'))
    model.add(Flatten())
    return model

def denc_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('tanh'))
    model.add(Flatten())
    return model

def discriminator_model():
    model = Sequential()
    model.add(Merge([denc_model(), dgen_model()], mode='concat'))
    model.add(Dense(input_dim=6400, units = 1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(g, d):
    d.trainable = False
    g_in = Input(shape=(100,))
    g_out = g(g_in)
    d_out = d([g_out, g_in])
    model = Model(inputs=g_in, outputs=d_out)
    return model

def encoder_containing_discriminator(e, d):
    d.trainable = False
    e_in = Input(shape=(28,28,1))
    e_out = e(e_in)
    d_out = d([e_in, e_out])
    model = Model(inputs=e_in, outputs=d_out)
    return model

'''def denc_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), strides=(1,1), padding='same', input_shape=(28,28,1)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (4,4), strides=(2,2)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (4,4), strides=(1,1)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (4,4), strides=(2,2)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (4,4), strides=(1,1)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    return model

def dgen_model():
    model = Sequential()
    model.add(Conv2D(512, (1,1), strides=(1,1), input_shape=(1,1,128)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, (1,1), strides=(1,1)))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Merge([denc_model(), dgen_model()], mode='concat'))
    model.add(Conv2D(1024, (1, 1), strides=(1, 1), input_shape=(1,1,1024)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Conv2D(1024, (1, 1), strides=(1, 1)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (1, 1), strides=(1, 1)))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model

# Define generator model, takes latent vector of dimension 100 and outputs image of size 28 x 28 x1
def encoder_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), strides=(1,1), padding='same', input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(64, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(128, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(256, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(512, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(512, (1, 1), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(128, (1,1), strides=(1,1)))
    model.add(Activation('tanh'))
    return model

# Define model with generator and discriminator together, this is how generator is trained
def generator_model():
    model = Sequential()
    model.add(Conv2DTranspose(256, (4,4), strides=(1,1), input_shape=(1,1,128)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(64, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(32, (5, 5), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(32, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(1, (3, 3), strides=(1, 1)))
    model.add(Activation('tanh'))
    return model

def generator_containing_discriminator(g,d):
    g_in = Input(shape=(1,1,128))
    g_out = g(g_in)
    d.trainable=False
    d_out = d([g_out, g_in])
    model = Model(inputs=g_in, outputs=d_out)
    return model

def encoder_containing_discriminator(e,d):
    e_in = Input(shape=(28,28,1))
    e_out = e(e_in)
    d.trainable=False
    d_out = d([e_in, e_out])
    model = Model(inputs=e_in, outputs=d_out)
    return model'''