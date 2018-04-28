from __future__ import print_function, division
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sd_models import encoder_model, generator_model
from common_models.common_models import autoencoder_model
from functions.auxiliary_funcs import save_models
from functions.visualization_funcs import save_reconstructions, save_latent_vis, plot_train_accuracy, plot_train_loss


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
image_path = 'Images/sd_basic_ae'
model_path = 'Models/sd_basic_ae'


# =====================================
# Load dataset
# =====================================

# Load dataset
X_train = np.loadtxt('Dataset/synthetic_dataset_x_train.txt', dtype=np.float32)
X_test = np.loadtxt('Dataset/synthetic_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/synthetic_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/synthetic_dataset_y_test.txt', dtype=np.int)

# Reshape to image format
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)


# =====================================
# Instantiate and compile models
# =====================================

# Instantiate models
generator = generator_model()
encoder = encoder_model()
autoencoder = autoencoder_model(encoder, generator)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# =====================================
# Train models
# =====================================

# Specify hyper-parameters for training
epochs = 100
batch_size = 128
patience = 10

# Specify callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [early_stopping, model_checkpoint]


# Train model
history = autoencoder.fit(X_train, X_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          callbacks=callbacks,
                          verbose=1)


# Replace current encoder and decoder models with that from the best save autoencoder
encoder = encoder_model()
decoder = generator_model()
autoencoder = autoencoder_model(encoder, decoder)
autoencoder.load_weights(model_path + '.h5')

# Save encoder and decoder models
save_models(path=model_path, encoder=encoder, generator=generator)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=False)

# Save latent space visualization
save_latent_vis(image_path, X_train, y_train, encoder, num_classes)

# Plot loss curves
plot_train_accuracy(image_path, history)
plot_train_loss(image_path, history)