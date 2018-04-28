from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Dropout, Activation, Reshape
from keras.models import Sequential


# Define models
def encoder_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    return model


def decoder_model():
    model = Sequential()
    model.add(Reshape((4,4,8), input_shape=(128,)))
    model.add(Conv2D(8, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    return model


def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    return model

def classifier_e_frozen_model(encoder):
    model = Sequential()
    encoder.trainable = False
    model.add(encoder)
    model.add(Flatten())
    #model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def classifier_e_trainable_model(encoder):
    model = Sequential()
    model.add(encoder)
    model.add(Flatten())
    #model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model