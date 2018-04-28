import keras
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from mnist_conv_ae_models import *
from keras.callbacks import EarlyStopping


# Set random seed for reproducibility
np.random.seed(1330)


## Load data set and change format

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Convert test labels to one hot format
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)


## Define and initialize classifier models

# Load pretrained encoder model and set non-trainable
pretrained_e = encoder_model()
pretrained_e.load_weights('encoder.h5')
pretrained_e.trainable=False

# Initialize classifier with e learned from autoencoder and frozen
mnist_classifier_pretrained_e = classifier_e_frozen_model(pretrained_e)

# Initialize classifier with randomly initialized e
random_e = encoder_model()
mnist_classifier_random_e = classifier_e_trainable_model(random_e)

# Print number of trainable and non-trainable parameters in both classifiers
trainable_count = int(
    np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e.trainable_weights)]))
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e.non_trainable_weights)]))

print('Classifier w/ Unsupervised Encoder + FC Layers')
print('Total parameters: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable parameters: {:,}'.format(trainable_count))
print('Non-trainable parameters: {:,}'.format(non_trainable_count))

# Print number of trainable and non-trainable parameters
trainable_count = int(
    np.sum([K.count_params(p) for p in set(mnist_classifier_random_e.trainable_weights)]))
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(mnist_classifier_random_e.non_trainable_weights)]))

print('\nClassifier w/ Random Encoder + FC Layers')
print('Total parameters: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable paramseter: {:,}'.format(trainable_count))
print('Non-trainable parameters: {:,}'.format(non_trainable_count))


## Pre-training

# Set hyperparameters and specify training details
batch_size = 100
epochs = 100
val_split = 1/5.
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')]


# Set number of labelled examples to investigate and no. of trainings to average test accuracy over
num_unlabelled = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 60000]
num_iterations = 5


## Create arrays to hold accuracy of classifiers
classifier_pretrained_acc = np.zeros(len(num_unlabelled))
classifier_random_acc = np.zeros(len(num_unlabelled))


# Try all no. of specified examples
for index, num in enumerate(num_unlabelled):

    print('Number of labelled examples: {:,}'.format(num))

    # Reset to zero for each
    pretrained_acc_sum = 0
    random_acc_sum = 0

    num_iterations = len(x_train) / num

    # Average test accuracy reading over num_iterations readings
    for iteration in range(num_iterations):

        # Reduce size of training sets
        reduced_x_train = x_train[(iteration * num) : ((iteration+1) * num), :, :, :]
        reduced_y_train = y_train_one_hot[(iteration * num) : ((iteration+1) * num), :]

        # Compile models
        mnist_classifier_pretrained_e.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])

        mnist_classifier_random_e.compile(loss=keras.losses.categorical_crossentropy,
                                          optimizer=keras.optimizers.Adadelta(),
                                          metrics=['accuracy'])


        # Train model with pretrained e
        mnist_classifier_pretrained_e.fit(reduced_x_train, reduced_y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                callbacks=callbacks,
                validation_split=val_split)

        # Add test accuracy to sum
        score = mnist_classifier_pretrained_e.evaluate(x_test, y_test_one_hot, verbose=0)
        pretrained_acc_sum += score[1]

        # Train model with random e
        mnist_classifier_random_e.fit(reduced_x_train, reduced_y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                callbacks=callbacks,
                validation_split=val_split)

        # Add test accuracy to sum
        score = mnist_classifier_random_e.evaluate(x_test, y_test_one_hot, verbose=0)
        random_acc_sum += score[1]

        ## Reinitialize both classifiers

        # Classifier with frozen e learned from autoencoder
        mnist_classifier_pretrained_e = classifier_e_frozen_model(pretrained_e)

        # Classifier with randomly initialized e
        random_e = encoder_model()
        mnist_classifier_random_e = classifier_e_free_model(random_e)


    # Record average classification accuracy for each no. of labelled examples
    classifier_pretrained_acc[index] = 100 * pretrained_acc_sum / num_iterations
    classifier_random_acc[index] = 100 * random_acc_sum / num_iterations


# Plot comparison graph
plt.plot(num_unlabelled, classifier_pretrained_acc, '-o', num_unlabelled, classifier_random_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['Pretrained Encoder', 'Randomly Initialized Encoder'], loc='lower right')
plt.grid()
plt.show()

# Plot for just pretrained network
plt.plot(num_unlabelled, classifier_pretrained_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training (Pretrained E')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.grid()
plt.show()

# Plot for just purely supervised network
plt.plot(num_unlabelled, classifier_random_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training (Random E)')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['Pretrained Encoder', 'Randomly Initialized Encoder'], loc='lower right')
plt.grid()
plt.show()