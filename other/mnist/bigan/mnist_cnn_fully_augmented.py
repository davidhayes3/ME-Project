'''Trains a simple convnet on the MNIST dataset.
Gets over 99% test accuracy after 12 epochs
3 to 4 seconds per epoch on a TitanX GPU.
'''
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np

batch_size = 128
num_classes = 10
epochs = 100
channels = 1

num_train_samples = 55000
num_val_samples = 5000


# Function to plot training loss curves
def plot_train_loss(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], -1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], -1, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, -1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, -1)
    input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile models
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Split training data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1 / 12., random_state=12345)

# Define augmentation process for images
data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Apply augmentation process to train and validation sets
train_batches = data_generator.flow(X_train, y_train, batch_size=batch_size)
val_batches = data_generator.flow(X_val, y_val, batch_size=batch_size)

# Specify callbacks
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0),
             ModelCheckpoint('mnist_faugmented_cnn.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

history = model.fit_generator(train_batches,
                              epochs=50,
                              steps_per_epoch=num_train_samples // batch_size,
                              validation_data=val_batches,
                              validation_steps=num_val_samples // batch_size,
                              callbacks=callbacks)

model.load_weights('mnist_faugmented_cnn.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
