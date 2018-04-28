import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functions.auxiliary_funcs import save_models
from functions.data_funcs import get_mnist
from functions.visualization_funcs import save_reconstructions, plot_train_loss
from mnist_mlp_models import encoder_model, generator_model
from common_models.common_models import latent_reconstructor_model
from keras.optimizers import Adam


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
image_path = 'Images/mnist_lr'
model_path = 'Models/mnist_lr'


# =====================================
# Load dataset
# =====================================

# Load MNIST data in range [-1,1]
(X_train, _), (X_test, y_test) = get_mnist(gan=True)

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
generator.load_weights('Models/mnist_gan_generator.h5')
generator.trainable = False
latent_regressor.compile(optimizer='SGD', loss='mse')


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

# Plot training curves
plot_train_loss(image_path, history)
