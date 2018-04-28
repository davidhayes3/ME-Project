import keras
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from common_models.classifier_models import classifier_e_frozen_model, classifier_e_trainable_model
from semi_supervised_comparison.cifar10_cnn.cifar10_models import deterministic_encoder_model
from functions.data_funcs import get_cifar10
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

# Number of labelled examples to investigate
num_unlabelled = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]
num_iterations = 5

# Path that containes pre-trained encoder
pretrained_encoder_path = 'cifar10_bigan_determ_encoder.h5'

# Paths to hold classifier models
classifier_pretrained_frozen_path = 'cifar10_pretrained_frozen_classifier.h5'
classifier_pretrained_trainable_path = 'cifar10_pretrained_trainable.h5'
classifier_random_path = 'cifar10_random.h5'


# =====================================
# Load data
# =====================================

(x_train, y_train), (x_test, y_test) = get_cifar10()

y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)


# =====================================
# Instantiate models
# =====================================

# Load frozen pretrained encoder model
pretrained_e_frozen = deterministic_encoder_model()
pretrained_e_frozen.load_weights(pretrained_encoder_path)
pretrained_e_frozen.trainable = False


# =====================================
# Training details
# =====================================

# Hyper-parameters and training specification for both models
epochs = 200
batch_size = 128
val_split = 1/5.
patience = 10

# Specify callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')

# Arrays to hold accuracy of classifiers
classifier_pretrained_frozen_acc = np.zeros(len(num_unlabelled))
classifier_pretrained_trainable_acc = np.zeros(len(num_unlabelled))
classifier_random_acc = np.zeros(len(num_unlabelled))


# =====================================
# Train models
# =====================================

# Loop through each quantity of enquiry
for index, num in enumerate(num_unlabelled):

    # Set each score to zero
    pretrained_frozen_score = 0
    pretrained_trainable_score = 0
    random_score = 0

    # Reduce size of training sets
    reduced_x_train = x_train[0:num, :, :, :]
    reduced_y_train = y_train_one_hot[0:num, :]


    # Average classification accuracy a number of random initializations
    for iteration in range(num_iterations):

        # Print details of no. of labelled examples and iteration number
        print('Labelled Examples: ' + str(num) + ', Iteration: ' + str(iteration+1) + '/' + str(num_iterations))

        # ----------------------------
        # Instantiate classifiers
        # ----------------------------

        # Classifier with e learned from autoencoder and frozen
        mnist_classifier_pretrained_e_frozen = classifier_e_frozen_model(pretrained_e_frozen)

        # Classifier with e learned from autoencoder and not frozen
        pretrained_e_trainable = deterministic_encoder_model()
        pretrained_e_trainable.load_weights(pretrained_encoder_path)
        mnist_classifier_pretrained_e_trainable = classifier_e_trainable_model(pretrained_e_trainable)

        # Classifier with randomly initialized e
        random_e = deterministic_encoder_model()
        mnist_classifier_random_e = classifier_e_trainable_model(random_e)


        # ----------------------------
        # Inspect trainable weights
        # ----------------------------

        # Print details of trainable and non-trainable weights of models
        if index == 0 and iteration == 0:

            # Print number of trainable and non-trainable parameters for each classifier

            trainable_count = int(
                np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e_frozen.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e_frozen.non_trainable_weights)]))

            print('Classifier w/ Frozen Pretrained Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


            trainable_count = int(
                np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e_trainable.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(mnist_classifier_pretrained_e_trainable.non_trainable_weights)]))

            print('\nClassifier w/ Trainable Pretrained Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable paramseter: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


            trainable_count = int(
                np.sum([K.count_params(p) for p in set(mnist_classifier_random_e.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(mnist_classifier_random_e.non_trainable_weights)]))

            print('\nClassifier w/ Random Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable paramseter: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


        # ----------------------------
        # Compile models
        # ----------------------------

        mnist_classifier_pretrained_e_frozen.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])

        mnist_classifier_pretrained_e_trainable.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])

        mnist_classifier_random_e.compile(loss=keras.losses.categorical_crossentropy,
                                          optimizer=keras.optimizers.Adadelta(),
                                          metrics=['accuracy'])

        # ----------------------------
        # Train classifiers
        # ----------------------------

        # Train classifier with frozen pretrained encoder
        model_checkpoint = ModelCheckpoint('classifier_1.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        mnist_classifier_pretrained_e_frozen.fit(reduced_x_train, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        mnist_classifier_pretrained_e_frozen.load_weights('classifier_1.h5')
        score = mnist_classifier_pretrained_e_frozen.evaluate(x_test, y_test_one_hot, verbose=0)
        pretrained_frozen_score += score[1]


        # Train classifier with trainable pretrained encoder
        model_checkpoint = ModelCheckpoint('classifier_2.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        mnist_classifier_pretrained_e_trainable.fit(reduced_x_train, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          callbacks=callbacks,
                                          shuffle=True,
                                          validation_split=val_split)

        mnist_classifier_pretrained_e_trainable.load_weights('classifier_2.h5')
        score = mnist_classifier_pretrained_e_trainable.evaluate(x_test, y_test_one_hot, verbose=0)
        pretrained_trainable_score += score[1]


        # Train classifier with randomly initialized encoder
        model_checkpoint = ModelCheckpoint('classifier_3.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        mnist_classifier_random_e.fit(reduced_x_train, reduced_y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=callbacks,
                                      shuffle=True,
                                      validation_split=val_split)

        mnist_classifier_random_e.load_weights('classifier_3.h5')
        score = mnist_classifier_random_e.evaluate(x_test, y_test_one_hot, verbose=0)
        random_score += score[1]


    # Record average classification accuracy for each no. of labelled examples
    classifier_pretrained_frozen_acc[index] = 100 * pretrained_frozen_score / num_iterations
    classifier_pretrained_trainable_acc[index] = 100 * pretrained_trainable_score / num_iterations
    classifier_random_acc[index] = 100 * random_score / num_iterations


# Print accuracies of classifiers on full training set
print('Classifer Accuracies\n')
print('Frozen Pretrained Encoder + FC Layers: ' + str(classifier_pretrained_frozen_acc[-1]) + '%')
print('Trainable Pretrained Encoder + FC Layers: ' + str(classifier_pretrained_trainable_acc[-1]) + '%')
print('Randomly Initialized Encoder + FC Layers: ' + str(classifier_random_acc[-1]) + '%')


# Save results to file
np.savetxt('classifier1.txt', classifier_pretrained_frozen_acc, fmt='%f')
np.savetxt('classifier2.txt', classifier_pretrained_trainable_acc, fmt='%f')
np.savetxt('classifier3.txt', classifier_random_acc, fmt='%f')


# =====================================
# Visualize results
# =====================================

# Plot comparison graph
plt.plot(num_unlabelled, classifier_pretrained_frozen_acc, '-o', num_unlabelled, classifier_pretrained_trainable_acc,
         '-o', num_unlabelled, classifier_random_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['Frozen Pretrained Encoder', 'Trainable Pretrained Encoder', 'Randomly Initialized Encoder'], loc='lower right')
plt.grid()
plt.savefig('cifar10_classifier_num_labels_compar.png')

# Plot for frozen pretrained network
plt.plot(num_unlabelled, classifier_pretrained_frozen_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training (Frozen Pretrained E')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.grid()
plt.savefig('cifar10_pretrained_frozen_acc.png')

# Plot for trainable pretrained network
plt.plot(num_unlabelled, classifier_pretrained_trainable_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training (Trainable Pretrained E)')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.grid()
plt.savefig('cifar10_pretrained_trainable_acc.png')

# Plot for supervised network
plt.plot(num_unlabelled, classifier_random_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training (Random E')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.grid()
plt.savefig('cifar10_random_acc.png')