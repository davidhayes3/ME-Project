import keras
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from mnist_conv_ae_models import *
from keras.callbacks import EarlyStopping
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

np.random.seed(1326) # for reproducibility

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

y_train_onehot = keras.utils.to_categorical(y_train, 10)
y_test_onehot = keras.utils.to_categorical(y_test, 10)

# Test features learned by encoder
# Load encoder and decoder models

pretrained_e = encoder_model()
pretrained_e.load_weights('encoder.h5')

# Build classifier using encoder from autoencoder, encoder is not trainable
mnist_classifier_pretrained_e = classifier_e_frozen_model(pretrained_e)

# Print number of trainable and non-trainable parameters
trainable_count = int(
    np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e.trainable_weights)]))
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e.non_trainable_weights)]))

print('Classifier w/ Unsupervised Encoder + FC Layers')
print('Total parameters: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable paramseter: {:,}'.format(trainable_count))
print('Non-trainable parameters: {:,}'.format(non_trainable_count))

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')]

# Hyperparameters for both models
batch_size=100
epochs=100
val_split = 1/5.
num=5000

# change size of training sets
x_train = x_train[0:num, :, :, :]
y_train_onehot = y_train_onehot[0:num]

# Train model
mnist_classifier_pretrained_e.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

mnist_classifier_pretrained_e.fit(x_train, y_train_onehot,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_split=val_split)

incorrects = np.nonzero(mnist_classifier_pretrained_e.predict_classes(x_test).reshape((-1,)) != y_test)
y_incorrects = y_test[incorrects]

# Plot frequency of incorrect labels
sns.countplot(x=y_incorrects, palette="Greens_d")
plt.ylabel('Number Predicted Incorrectly')
plt.xlabel('MNIST Digit')
plt.show()

