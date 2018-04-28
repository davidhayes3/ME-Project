import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functions.auxiliary_funcs import save_models
from functions.data_funcs import get_mnist
from functions.visualization_funcs import save_reconstructions, plot_train_accuracy, plot_train_loss
from mnist_mlp_models import encoder_model, generator_model
from common_models.common_models import autoencoder_model


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
image_path = 'Images/mnist_basic_ae'
model_path = 'Models/mnist_basic_ae'


# =====================================
# Load dataset
# =====================================

(X_train, _), (X_test, y_test) = get_mnist()


# =====================================
# Instantiate and compile models
# =====================================

encoder = encoder_model()
generator = generator_model()
autoencoder = autoencoder_model(encoder, generator)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 50
batch_size = 128
patience = 5

# Specify callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                   mode='min')
callbacks = [early_stopping, model_checkpoint]


# Train model
history = autoencoder.fit(X_train, X_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_split=1/12.,
                          callbacks=callbacks,
                          verbose=1)


# Replace current encoder and decoder models with that from the best save autoencoder
encoder = encoder_model()
generator = generator_model()
autoencoder = autoencoder_model(encoder, generator)
autoencoder.load_weights(model_path + '.h5')

# Save encoder and decoder models
save_models(path=model_path, encoder=encoder, generator=generator)


# =====================================
# Visualizations
# =====================================

# Save reconstructions of test images
save_reconstructions(image_path, num_classes, X_test, y_test, generator, encoder, img_rows, img_cols, channels, color=False)

# Plot loss curves
plot_train_accuracy(image_path, history)
plot_train_loss(image_path, history)