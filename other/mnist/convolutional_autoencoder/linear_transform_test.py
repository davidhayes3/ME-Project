from scipy.fftpack import dct, idct
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation, Reshape, Input
from keras.models import Sequential, Model
import keras.utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import pywt


# Define models
def encoder_model():
    z = Input(shape=(28,28,1))

    #z = dct(dct(z.T, norm='ortho').T, norm='ortho')

    x = Conv2D(16, kernel_size=(3, 3), padding='same')(z)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)

    return Model(z, x)


def decoder_model():
    x = Input(shape=(128,))

    z = Reshape((4,4,8))(x)

    z = Conv2D(8, kernel_size=(3, 3), padding='same')(z)
    z = Activation('relu')(z)
    z = UpSampling2D((2,2))(z)
    z = Conv2D(8, (3, 3), padding='same')(z)
    z = Activation('relu')(z)
    z = UpSampling2D((2, 2))(z)
    z = Conv2D(16, (3, 3))(z)
    z = Activation('relu')(z)
    z = UpSampling2D((2, 2))(z)
    z = Conv2D(1, (3, 3), padding='same')(z)
    z = Activation('sigmoid')(z)

    #z = idct(idct(z.T, norm='ortho').T, norm='ortho')

    return Model(x, z)


def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    return model



np.random.seed(1337) # for reproducibility

# Load dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

dct_x_train = dct(x_train)
dct_x_test = dct(x_test)
dwt_x_train = pywt.dwt2(x_train, wavelet='coif1')
print(dct_x_train.shape)



plt.imshow(x_train[0].reshape(28, 28))
plt.show()
plt.imshow(dct_x_train[0].reshape(28, 28))
plt.show()
plt.imshow(dwt_x_train[0].reshape(28, 28))
plt.show()



y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Create models for encoder, decoder and combined autoencoder
e = encoder_model()
d = decoder_model()
autoencoder = autoencoder_model(e, d)


# Specify loss function and optimizer for autoencoder
#autoencoder.compile(optimizer='adam', loss='mse',  metrics=['accuracy'])
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',  metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
            TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=5, write_graph=True,
                        write_images=True)]

history = autoencoder.fit(dct_x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_split = 1/12.,
                callbacks=callbacks,
                verbose=1
            )


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