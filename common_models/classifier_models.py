from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation


def classifier_e_frozen_model(encoder):
    model = Sequential()

    encoder.trainable = False

    model.add(encoder)
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def classifier_e_trainable_model(encoder):
    model = Sequential()

    model.add(encoder)
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def mnist_classifier_e_trainable_model(encoder):
    model = Sequential()

    model.add(encoder)

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def mnist_classifier_e_frozen_model(encoder):
    model = Sequential()

    encoder.trainable = False
    model.add(encoder)

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model