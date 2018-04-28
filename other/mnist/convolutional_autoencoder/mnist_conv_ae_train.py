from mnist_conv_ae_models import *
import keras.utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K

np.random.seed(1337) # for reproducibility

# Load dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Create models for encoder, decoder and combined autoencoder
e = encoder_model()
d = decoder_model()
autoencoder = autoencoder_model(e, d)
print(e.count_params(), d.count_params(), autoencoder.count_params())


# Specify loss function and optimizer for autoencoder
#autoencoder.compile(optimizer='adam', loss='mse',  metrics=['accuracy'])
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',  metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
            TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=5, write_graph=True,
                        write_images=True),
            ModelCheckpoint('mnist_conv_autoencoder.h5', monitor='val_loss', save_best_only=True, verbose=0)
]

history = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_split = 1/12.,
                callbacks=callbacks,
                verbose=1
            )


# Save encoder and decoder models
e.save_weights('mnist_conv_ae_encoder.h5', True)
d.save_weights('mnist_conv_ae_decoder.h5', True)

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', ' Validation'], loc='lower right')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# Reconstruct images based on learned autencoder
recon_imgs = autoencoder.predict(x_test)


# Plot reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()