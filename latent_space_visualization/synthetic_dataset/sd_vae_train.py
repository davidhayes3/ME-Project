from __future__ import print_function, division
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from common_models.common_models import vae_encoder_sampling_model, vae_model
from sd_models import vae_encoder_model, generator_model
from functions.auxiliary_funcs import save_models
from functions.visualization_funcs import save_reconstructions, save_latent_vis, plot_train_loss
import numpy as np


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_dim = 4
img_rows = 2
img_cols = 2
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 2
num_classes = 16
epsilon_std = 0.05
image_path = 'Images/sd_vae'
model_path = 'Models/sd_vae'


# =====================================
# Load dataset
# =====================================

# Load dataset
X_train = np.loadtxt('Dataset/synthetic_dataset_x_train.txt', dtype=np.float32)
X_test = np.loadtxt('Dataset/synthetic_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/synthetic_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/synthetic_dataset_y_test.txt', dtype=np.int)

# Reshape data to image format
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)


# =====================================
# Instantiate and compile models
# =====================================

# Define sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon


# Instantiate models
encoder = vae_encoder_model()
generator = generator_model()


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
epochs = 20
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
                  validation_data=(X_test, None))


# Replace current encoder and decoder models with that from the best save autoencoder
encoder = vae_encoder_model()
sampled_encoder = vae_encoder_sampling_model(encoder, latent_dim, img_shape, epsilon_std)
generator = generator_model()
vae = vae_model(sampled_encoder, generator, img_shape)
vae.load_weights(model_path + '.h5')

# Save encoder and decoder models
save_models(path=model_path, encoder=encoder, generator=generator)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, sampled_encoder, img_rows, img_cols, channels, color=False)

# Save latent space visualization
save_latent_vis(image_path, X_train, y_train, sampled_encoder, num_classes)

# Plot training curves
plot_train_loss(image_path, history)