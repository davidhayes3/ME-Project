import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from cifar10_models import deterministic_encoder_model, vae_encoder_model
from common_models.classifier_models import classifier_e_trainable_model, classifier_e_frozen_model
from common_models.common_models import vae_encoder_sampling_model
from functions.data_funcs import get_cifar10


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
model_path = 'Models/cifar10'
results_path = 'Results/cifar10'


# =====================================
# Load data
# =====================================

# Load pre-processed CIFAR10 data
(X_train, y_train), (X_test, y_test) = get_cifar10()


# Label data is same for both
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)


# =====================================
# Instantiate models
# =====================================

# Load encoders
basic_ae = deterministic_encoder_model()
dae = deterministic_encoder_model()
aae = deterministic_encoder_model()
vae_encoder = vae_encoder_model()
bigan = deterministic_encoder_model()


# Load saved weights
basic_ae.load_weights(model_path + '_basic_ae_encoder.h5')
dae.load_weights(model_path + '_dae_encoder.h5')
aae.load_weights(model_path + '_aae_encoder.h5')
vae_encoder.load_weights(model_path + '_vae_encoder.h5')
vae = vae_encoder_sampling_model(vae_encoder, latent_dim, img_shape, 0.05)
bigan.load_weights(model_path + '_bigan_determ_encoder.h5')


# Freeze the parameters of all encoders
basic_ae.trainable = False
dae.trainable = False
aae.trainable = False
vae.trainable = False
bigan.trainable = False


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 100
batch_size = 128
val_split = 1/5.
patience = 10

# Specify optimizer for classifier training
optimizer = keras.optimizers.Adadelta()

# Specify training stop criterion
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')

# Number of labelled examples to investigate
num_unlabelled = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]

# Number of random initializations of FC layers for each value in num_unlabelled
num_initializations = 5

# Arrays to hold accuracy of classifiers
classifier1_acc = np.zeros(len(num_unlabelled))
classifier2_acc = np.zeros(len(num_unlabelled))
classifier3_acc = np.zeros(len(num_unlabelled))
classifier4_acc = np.zeros(len(num_unlabelled))
classifier5_acc = np.zeros(len(num_unlabelled))
classifier6_acc = np.zeros(len(num_unlabelled))


# Train classifiers for each number of unlabeled examples
for index, num in enumerate(num_unlabelled):

    # Reset classifier scores to zero
    classifier1_score = 0
    classifier2_score = 0
    classifier3_score = 0
    classifier4_score = 0
    classifier5_score = 0
    classifier6_score = 0

    # Reduce size of training sets
    reduced_x_train = X_train[0:num, :, :, :]
    reduced_y_train = y_train_one_hot[0:num, :]

    # Average classification accuracy over num_iterations readings
    for initialization in range(num_initializations):

        # Print details of no. of labelled examples and iteration number
        print('Labelled Examples: ' + str(num) + ', Iteration: ' + str(initialization+1) + '/' + str(num_initializations))

        # Instantiate classfiers to be trained
        classifier1 = classifier_e_frozen_model(basic_ae)
        classifier2 = classifier_e_frozen_model(dae)
        classifier3 = classifier_e_frozen_model(aae)
        classifier4 = classifier_e_frozen_model(vae)
        classifier5 = classifier_e_frozen_model(bigan)
        cnn = deterministic_encoder_model()
        classifier6 = classifier_e_trainable_model(cnn)

        # Compile models
        for classifier in (classifier1, classifier2, classifier3, classifier4, classifier5, classifier6):
            classifier.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=optimizer,
                               metrics=['accuracy'])


        # =====================================
        # Train classifiers
        # =====================================

        # Classifier 1
        model_checkpoint = ModelCheckpoint(model_path + '_classifier_1.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier1.fit(reduced_x_train, reduced_y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks,
                        validation_split=val_split)

        classifier1.load_weights(model_path + '_classifier_1.h5')
        score = classifier1.evaluate(X_test, y_test_one_hot, verbose=0)
        classifier1_score += score[1]


        # Classifier 2
        model_checkpoint = ModelCheckpoint(model_path + '_classifier_2.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier2.fit(reduced_x_train, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier2.load_weights(model_path + '_classifier_2.h5')
        score = classifier2.evaluate(X_test, y_test_one_hot, verbose=0)
        classifier2_score += score[1]


        # Classifier 3
        model_checkpoint = ModelCheckpoint(model_path + '_classifier_3.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier3.fit(reduced_x_train, reduced_y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks,
                        validation_split=val_split)

        classifier3.load_weights(model_path + '_classifier_3.h5')
        score = classifier3.evaluate(X_test, y_test_one_hot, verbose=0)
        classifier3_score += score[1]


        # Classifier 4
        model_checkpoint = ModelCheckpoint(model_path + '_classifier_4.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier4.fit(reduced_x_train, reduced_y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks,
                        validation_split=val_split)

        classifier4.load_weights(model_path + '_classifier_4.h5')
        score = classifier4.evaluate(X_test, y_test_one_hot, verbose=0)
        classifier4_score += score[1]


        # Classifier 5
        model_checkpoint = ModelCheckpoint(model_path + '_classifier_5.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier5.fit(reduced_x_train, reduced_y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks,
                        validation_split=val_split)

        classifier5.load_weights(model_path + '_classifier_5.h5')
        score = classifier5.evaluate(X_test, y_test_one_hot, verbose=0)
        classifier5_score += score[1]


        # Classifier 6
        model_checkpoint = ModelCheckpoint(model_path + '_classifier_6.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier6.fit(reduced_x_train, reduced_y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks,
                        validation_split=val_split)

        classifier6.load_weights(model_path + '_classifier_6.h5')
        score = classifier6.evaluate(X_test, y_test_one_hot, verbose=0)
        classifier6_score += score[1]


    # Record average classification accuracy for each no. of labelled examples
    classifier1_acc[index] = 100 * classifier1_score / num_initializations
    classifier2_acc[index] = 100 * classifier2_score / num_initializations
    classifier3_acc[index] = 100 * classifier3_score / num_initializations
    classifier4_acc[index] = 100 * classifier4_score / num_initializations
    classifier5_acc[index] = 100 * classifier5_score / num_initializations
    classifier6_acc[index] = 100 * classifier6_score / num_initializations


    # Save results for all classifiers to file
    np.savetxt('Results/classifier1.txt', classifier1_acc, fmt='%f')
    np.savetxt('Results/classifier2.txt', classifier2_acc, fmt='%f')
    np.savetxt('Results/classifier3.txt', classifier3_acc, fmt='%f')
    np.savetxt('Results/classifier4.txt', classifier4_acc, fmt='%f')
    np.savetxt('Results/classifier5.txt', classifier5_acc, fmt='%f')
    np.savetxt('Results/classifier6.txt', classifier6_acc, fmt='%f')


# Print accuracies
print(classifier1_acc)
print(classifier2_acc)
print(classifier3_acc)
print(classifier4_acc)
print(classifier5_acc)
print(classifier6_acc)