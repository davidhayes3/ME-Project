import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functions.auxiliary_funcs import save_models
from functions.data_funcs import get_cifar10
from functions.visualization_funcs import save_reconstructions, plot_train_accuracy, plot_train_loss
from cifar10_models import deterministic_encoder_model, generator_model, autoencoder_model


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 64
num_classes = 10
image_path = 'Images/cifar10_dae'
model_path = 'Models/cifar10_dae'


# =====================================
# Load dataset
# =====================================

(X_train, _), (X_test, y_test) = get_cifar10()

# Corrupt data with noise
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(0., 1, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(0., 1, size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)


# =====================================
# Instantiate and compile models
# =====================================

encoder = deterministic_encoder_model()
generator = generator_model()
generator.load_weights('cifar10_gan_generator')
autoencoder = autoencoder_model(encoder, generator)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])


# =====================================
# Train models
# =====================================

# Specify hyper-parameters for training
epochs = 100
batch_size = 128
patience = 5

# Specify callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [early_stopping, model_checkpoint]


# Train model
history = autoencoder.fit(X_train_noisy, X_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_split=0.1,
                          callbacks=callbacks,
                          verbose=1)


# Replace current encoder and decoder models with that from the best save autoencoder
encoder = deterministic_encoder_model()
decoder = generator_model()
autoencoder = autoencoder_model(encoder, decoder)
autoencoder.load_weights(model_path + '.h5')

# Save encoder and decoder models
save_models(path=model_path, encoder=encoder, generator=generator)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=True)

# Plot loss curves
plot_train_loss(image_path, history)
plot_train_accuracy(image_path, history)