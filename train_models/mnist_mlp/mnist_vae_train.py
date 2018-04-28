from __future__ import print_function
import numpy as np
from functions.data_funcs import get_mnist
from functions.visualization_funcs import plot_train_loss, save_reconstructions
from functions.auxiliary_funcs import save_models
from mnist_mlp_models import vae_encoder_model, generator_model
from common_models.common_models import vae_model, vae_encoder_sampling_model
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.layers import Input, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
num_classes = 10
image_path = 'Images/mnist_vae'
model_path = 'Models/mnist_vae'
epsilon_std = 0.05


# =====================================
# Load dataset
# =====================================

(X_train, y_train), (X_test, y_test) = get_mnist()


# =====================================
# Instantiate and compile models
# =====================================

# Instantiate models
encoder = vae_encoder_model()
generator = generator_model()


# Define sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon


# Define VAE model
x = Input(shape=img_shape)

z_mean, z_log_var = encoder(x)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
recon_x = generator(z)

vae = Model(x, recon_x)


# Define VAE loss and compile model
xent_loss = np.prod(img_shape) * K.mean(metrics.binary_crossentropy(x, recon_x))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop', loss=None)


# =====================================
# Train models
# =====================================

# Specify training hyper-parameters
epochs = 50
batch_size = 128
patience = 10

# Specify callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(filepath=model_path+'.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                   mode='min')
callbacks = [early_stopping, model_checkpoint]


# Train model
history = vae.fit(X_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  callbacks=callbacks,
                  validation_split=1/12.)


# Replace current encoder and decoder models with that from the best save autoencoder
stochastic_encoder = vae_encoder_model()
encoder = vae_encoder_sampling_model(stochastic_encoder, latent_dim, img_shape, epsilon_std)
generator = generator_model()
vae = vae_model(encoder, generator, img_shape)
vae.load_weights(model_path + '.h5')

# Save encoder and decoder models
save_models(path=model_path, encoder=encoder, generator=generator)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=False)

# Plot training curves
plot_train_loss(image_path, history)