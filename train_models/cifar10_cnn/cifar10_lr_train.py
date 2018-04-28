import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from functions.auxiliary_funcs import save_models
from functions.data_funcs import get_cifar10
from functions.visualization_funcs import save_reconstructions, plot_train_loss
from cifar10_models import deterministic_encoder_model, generator_model
from common_models.common_models import latent_reconstructor_model


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
image_path = 'Images/cifar10_lr'
model_path = 'Models/cifar10_lr'


# =====================================
# Load dataset
# =====================================

(X_train, _), (X_test, y_test) = get_cifar10()

z_train = np.random.normal(size=(X_train.shape[0], latent_dim))
z_test = np.random.normal(size=(X_test.shape[0], latent_dim))


# =====================================
# Instantiate and compile models
# =====================================

# Instanstiate models
encoder = deterministic_encoder_model()
generator = generator_model()
generator.load_weights('Models/cifar10_bigan_determ_generator.h5')
generator.trainable = False
latent_regressor = latent_reconstructor_model(generator, encoder)

# Specify optimizer
lr = 0.0002
beta_1 = 0.5
optimizer = Adam(lr=lr, beta_1=beta_1)

# Compile latent regressor
latent_regressor.compile(optimizer=optimizer, loss='mse')


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 100
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
encoder = deterministic_encoder_model()
latent_reconstructor = latent_reconstructor_model(decoder, encoder)
latent_reconstructor.load_weights(model_path + '.h5')

# Save encoder weights
save_models(path=model_path, encoder=encoder)


# =====================================
# Visualization
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=True)

# Plot training curves
plot_train_loss(image_path, history)