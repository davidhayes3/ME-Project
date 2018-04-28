from __future__ import print_function, division
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sd_models import encoder_model, generator_model
from common_models.common_models import latent_reconstructor_model
from functions.auxiliary_funcs import save_models
from functions.visualization_funcs import save_reconstructions, save_latent_vis, plot_train_loss


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
image_path = 'Images/sd_lr'
model_path = 'Models/sd_lr'


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

# Normalize data to (-1,1)
X_train = (X_train - 0.5) / 0.5
X_test = (X_test - 0.5) / 0.5

# Samples from prior used to train latent regressor
z_train = np.random.normal(size=(X_train.shape[0], latent_dim))
z_test = np.random.normal(size=(X_test.shape[0], latent_dim))


# =====================================
# Instantiate and compile models
# =====================================

# Instanstiate models
encoder = encoder_model()
generator = generator_model(gan=True)
latent_regressor = latent_reconstructor_model(generator, encoder)

# Compile latent regressor
generator.load_weights('Models/sd_gan_generator.h5')
generator.trainable = False
latent_regressor.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 50
batch_size = 128
patience = 5

# Specify training stopping criterion
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                   mode='min')
callbacks = [early_stopping, model_checkpoint]


# Train model
history = latent_regressor.fit(z_train, z_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               shuffle=True,
                               validation_data=(z_test, z_test),
                               callbacks=callbacks,
                               verbose=1)


# Replace current encoder and decoder models with that from the saved best autoencoder
decoder = generator_model()
encoder = encoder_model()
latent_reconstructor = latent_reconstructor_model(decoder, encoder)
latent_reconstructor.load_weights(model_path + '.h5')

# Save encoder weights
save_models(path=model_path, encoder=encoder)


# =====================================
# Visualization
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=False)

# Save latent visualization
save_latent_vis(image_path, X_train, y_train, encoder, num_classes)

# Plot training curves
plot_train_loss(image_path, history)
