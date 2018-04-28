import keras
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from common_models.classifier_models import classifier_e_frozen_model, classifier_e_trainable_model
from train_models.cifar10_cnn.cifar10_models import deterministic_encoder_model
from functions.data_funcs import get_cifar10
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

# Number of labelled examples to investigate
num_unlabelled = [200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]
num_iterations = 5
num_classes = 10

# Path that containes pre-trained encoder
pretrained_encoder_path = 'cifar10_bigan_determ_encoder.h5'

# Paths to hold classifier models
classifier_pretrained_path = 'cifar10_pretrained_classifier.h5'
classifier_pretrained_aug_path = 'cifar10_pretrained_aug_classifier.h5'
classifier_random_path = 'cifar10_random.h5'
classifier_random_aug_path = 'cifar10_random_aug.h5'


# =====================================
# Load data
# =====================================

(X_train, y_train), (X_test, y_test) = get_cifar10()

y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)


# =====================================
# Define augmentation
# =====================================

datagen = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')


# =====================================
# Instantiate models
# =====================================

# Load frozen pretrained encoder model
pretrained_e = deterministic_encoder_model()
pretrained_e_one_layer_trainable = deterministic_encoder_model()
pretrained_e_trainable = deterministic_encoder_model()

# Load weights
pretrained_e.load_weights(pretrained_encoder_path)


# =====================================
# Training details
# =====================================

# Hyper-parameters and training specification for both models
epochs = 100
aug_epochs = 50
batch_size = 128
val_split = 1/5.
patience = 10

# Specify callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')

# Arrays to hold accuracy of classifiers
classifier_pretrained_acc = np.zeros(len(num_unlabelled))
classifier_pretrained_aug_acc = np.zeros(len(num_unlabelled))
classifier_pretrained_lastconv_acc = np.zeros(len(num_unlabelled))
classifier_pretrained_lastconv_aug_acc = np.zeros(len(num_unlabelled))
classifier_random_acc = np.zeros(len(num_unlabelled))
classifier_random_aug_acc = np.zeros(len(num_unlabelled))
classifier_pretrained_trainable_acc = np.zeros(len(num_unlabelled))
classifier_pretrained_trainable_aug_acc = np.zeros(len(num_unlabelled))


# =====================================
# Train models
# =====================================

# Loop through each quantity of enquiry
for index, num in enumerate(num_unlabelled):

    # Set each score to zero
    pretrained_score = 0
    pretrained_aug_score = 0
    pretrained_lastconv_score = 0
    pretrained_lastconv_aug_score = 0
    random_score = 0
    random_aug_score = 0
    pretrained_trainable_score = 0
    pretrained_trainable_aug_score = 0

    # Reduce size of training sets
    reduced_x_train = X_train[0:num, :, :, :]
    reduced_y_train = y_train_one_hot[0:num, :]

    # fit the dataget
    datagen.fit(reduced_x_train)


    # Average classification accuracy a number of random initializations
    for iteration in range(num_iterations):

        # Print details of no. of labelled examples and iteration number
        print('Labelled Examples: ' + str(num) + ', Iteration: ' + str(iteration+1) + '/' + str(num_iterations))


        # ----------------------------
        # Instantiate classifiers
        # ----------------------------

        # Classifiers with encoder learned from autoencoder and frozen
        classifier_pretrained = classifier_e_frozen_model(pretrained_e)
        classifier_pretrained_aug = classifier_e_frozen_model(pretrained_e)

        # Classifiers with encoder learned from autoencoder and frozen (except last conv layer)
        pretrained_e_one_layer_trainable.load_weights(pretrained_encoder_path)
        # Set all layers to be non-trainable except last conv
        classifier_pretrained_lastconv = classifier_e_trainable_model(pretrained_e_one_layer_trainable)
        classifier_pretrained_lastconv_aug = classifier_e_trainable_model(pretrained_e_one_layer_trainable)

        for i, layer in enumerate(pretrained_e_one_layer_trainable.layers):
            if i != 17:
                if i != 19:
                    layer.trainable = False

        # Classifier with randomly initialized encoder
        random_e = deterministic_encoder_model()
        classifier_random = classifier_e_trainable_model(random_e)
        classifier_random_aug = classifier_e_trainable_model(random_e)

        # Classifier with trainable pre-trained encoder
        pretrained_e_trainable.load_weights(pretrained_encoder_path)
        classifier_pretrained_trainable = classifier_e_trainable_model(pretrained_e_trainable)
        classifier_pretrained_trainable_aug = classifier_e_trainable_model(pretrained_e_trainable)



        # ----------------------------
        # Inspect trainable weights
        # ----------------------------

        # Print details of trainable and non-trainable weights of models
        if index == 0 and iteration == 0:

            # Print number of trainable and non-trainable parameters for each classifier

            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained.non_trainable_weights)]))

            print('\nClassifier w/ Frozen Pretrained Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_aug.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_aug.non_trainable_weights)]))

            print('\nClassifier w/ Frozen Pretrained Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_lastconv.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_lastconv.non_trainable_weights)]))

            print('\nClassifier w/ Trainable Pretrained Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_lastconv_aug.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_lastconv_aug.non_trainable_weights)]))

            print('\nClassifier w/ Trainable Pretrained Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_random.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_random.non_trainable_weights)]))

            print('\nClassifier w/ Random Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_random_aug.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_random_aug.non_trainable_weights)]))

            print('\nClassifier w/ Random Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))

            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_trainable.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_trainable.non_trainable_weights)]))

            print('\nClassifier w/ Fully Trainable Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))

            trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_trainable_aug.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(classifier_pretrained_trainable_aug.non_trainable_weights)]))

            print('\nClassifier w/ Fully Trainable Encoder + FC Layers')
            print('Total parameters: ' + str(trainable_count + non_trainable_count))
            print('Trainable parameters: ' + str(trainable_count))
            print('Non-trainable parameters: ' + str(non_trainable_count))


        # ----------------------------
        # Compile models
        # ----------------------------

        classifier_pretrained.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])

        classifier_pretrained_aug.compile(loss=keras.losses.categorical_crossentropy,
                                                  optimizer=keras.optimizers.Adadelta(),
                                                  metrics=['accuracy'])

        classifier_pretrained_lastconv.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])

        classifier_pretrained_lastconv_aug.compile(loss=keras.losses.categorical_crossentropy,
                                                  optimizer=keras.optimizers.Adadelta(),
                                                  metrics=['accuracy'])

        classifier_random.compile(loss=keras.losses.categorical_crossentropy,
                                          optimizer=keras.optimizers.Adadelta(),
                                          metrics=['accuracy'])

        classifier_random_aug.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])

        classifier_pretrained_trainable.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])

        classifier_pretrained_trainable_aug.compile(loss=keras.losses.categorical_crossentropy,
                                                  optimizer=keras.optimizers.Adadelta(),
                                                  metrics=['accuracy'])


        # ----------------------------
        # Train classifiers
        # ----------------------------

        # Train classifier with frozen pretrained encoder
        model_checkpoint = ModelCheckpoint('classifier_1.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier_pretrained.fit(reduced_x_train, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier_pretrained.load_weights('classifier_1.h5')
        score = classifier_pretrained.evaluate(X_test, y_test_one_hot, verbose=0)
        pretrained_score += score[1]


        # Train previous classifier with augmentation
        classifier_pretrained_aug.load_weights('classifier_1.h5')

        train_batches = datagen.flow(reduced_x_train, reduced_y_train, batch_size=batch_size)

        classifier_pretrained_aug.fit_generator(train_batches,
                                                epochs=aug_epochs,
                                                steps_per_epoch=reduced_x_train.shape[0]//batch_size)

        classifier_pretrained_aug.save_weights('classifier_2.h5')
        score = classifier_pretrained_aug.evaluate(X_test, y_test_one_hot, batch_size=batch_size, verbose=1)
        pretrained_aug_score += score[1]


        # Train classifier with frozen pretrained encoder (last conv layer trainable)
        model_checkpoint = ModelCheckpoint('classifier_3.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier_pretrained_lastconv.fit(reduced_x_train, reduced_y_train,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=callbacks,
                                  validation_split=val_split)

        classifier_pretrained_lastconv.load_weights('classifier_3.h5')
        score = classifier_pretrained_lastconv.evaluate(X_test, y_test_one_hot, verbose=0)
        pretrained_lastconv_score += score[1]

        # Train previous classifier with augmentation
        classifier_pretrained_lastconv_aug.load_weights('classifier_3.h5')

        train_batches = datagen.flow(reduced_x_train, reduced_y_train, batch_size=batch_size)

        classifier_pretrained_lastconv_aug.fit_generator(train_batches,
                                                epochs=aug_epochs,
                                                steps_per_epoch=reduced_x_train.shape[0] // batch_size)

        classifier_pretrained_lastconv_aug.save_weights('classifier_4.h5')
        score = classifier_pretrained_lastconv_aug.evaluate(X_test, y_test_one_hot, batch_size=batch_size, verbose=1)
        pretrained_lastconv_aug_score += score[1]


        # Train classifier with randomly initialized encoder
        model_checkpoint = ModelCheckpoint('classifier_5.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier_random.fit(reduced_x_train, reduced_y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=callbacks,
                                      shuffle=True,
                                      validation_split=val_split)

        classifier_random.load_weights('classifier_5.h5')
        score = classifier_random.evaluate(X_test, y_test_one_hot, verbose=0)
        random_score += score[1]


        # Train previous classifier with augmentation
        classifier_random_aug.load_weights('classifier_5.h5')

        train_batches = datagen.flow(reduced_x_train, reduced_y_train, batch_size=batch_size)

        classifier_random_aug.fit_generator(train_batches,
                                            epochs=aug_epochs,
                                            steps_per_epoch=reduced_x_train.shape[0]//batch_size)

        classifier_random_aug.save_weights('classifier_6.h5')
        score = classifier_random_aug.evaluate(X_test, y_test_one_hot, batch_size=batch_size, verbose=1)
        random_aug_score += score[1]



        # Train classifier with frozen pretrained encoder (last conv layer trainable)
        model_checkpoint = ModelCheckpoint('classifier_7.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier_pretrained_trainable.fit(reduced_x_train, reduced_y_train,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=callbacks,
                                  validation_split=val_split)

        classifier_pretrained_trainable.load_weights('classifier_7.h5')
        score = classifier_pretrained_trainable.evaluate(X_test, y_test_one_hot, verbose=0)
        pretrained_trainable_score += score[1]

        # Train previous classifier with augmentation
        classifier_pretrained_trainable_aug.load_weights('classifier_7.h5')

        train_batches = datagen.flow(reduced_x_train, reduced_y_train, batch_size=batch_size)

        classifier_pretrained_trainable_aug.fit_generator(train_batches,
                                                epochs=aug_epochs,
                                                steps_per_epoch=reduced_x_train.shape[0] // batch_size)

        classifier_pretrained_trainable_aug.save_weights('classifier_8.h5')
        score = classifier_pretrained_trainable_aug.evaluate(X_test, y_test_one_hot, batch_size=batch_size, verbose=1)
        pretrained_trainable_aug_score += score[1]



    # Record average classification accuracy for each no. of labelled examples
    classifier_pretrained_acc[index] = 100 * pretrained_score / num_iterations
    classifier_pretrained_aug_acc[index] = 100 * pretrained_aug_score / num_iterations
    classifier_pretrained_lastconv_acc[index] = 100 * pretrained_lastconv_score / num_iterations
    classifier_pretrained_lastconv_aug_acc[index] = 100 * pretrained_lastconv_aug_score / num_iterations
    classifier_random_acc[index] = 100 * random_score / num_iterations
    classifier_random_aug_acc[index] = 100 * random_aug_score / num_iterations
    classifier_pretrained_trainable_acc[index] = 100 * pretrained_trainable_score / num_iterations
    classifier_pretrained_trainable_aug_acc[index] = 100 * pretrained_trainable_aug_score / num_iterations


    # Save results to file
    np.savetxt('Results/classifier1.txt', classifier_pretrained_acc, fmt='%f')
    np.savetxt('Results/classifier2.txt', classifier_pretrained_aug_acc, fmt='%f')
    np.savetxt('Results/classifier3.txt', classifier_pretrained_lastconv_acc, fmt='%f')
    np.savetxt('Results/classifier4.txt', classifier_pretrained_lastconv_aug_acc, fmt='%f')
    np.savetxt('Results/classifier5.txt', classifier_random_acc, fmt='%f')
    np.savetxt('Results/classifier6.txt', classifier_random_aug_acc, fmt='%f')
    np.savetxt('Results/classifier7.txt', classifier_pretrained_trainable_acc, fmt='%f')
    np.savetxt('Results/classifier8.txt', classifier_pretrained_trainable_aug_acc, fmt='%f')


# =====================================
# Visualize results
# =====================================

# Plot comparison graph
plt.figure()
plt.plot(num_unlabelled, classifier_pretrained_acc, '-o', num_unlabelled, classifier_pretrained_aug_acc,
         '-o', num_unlabelled, classifier_random_aug_acc, '-o')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['BiGAN Encoder + No Augmentation', 'BiGAN Encoder + Augmentation',
            'Randomly Initialized Encoder + Augmentation'], loc='lower right')
plt.grid()
plt.savefig('cifar10_bigan_aug_compar.png')


# Plot comparison graph
plt.figure()
plt.plot(num_unlabelled, classifier_pretrained_acc, '-o', num_unlabelled, classifier_pretrained_aug_acc,
         '-o', num_unlabelled, classifier_pretrained_lastconv_acc, '-o', num_unlabelled, classifier_pretrained_lastconv_aug_acc, '-o',
         num_unlabelled, classifier_pretrained_trainable_acc, '-o', num_unlabelled, classifier_pretrained_trainable_aug_acc, '-o',
         num_unlabelled, classifier_random_aug_acc, '-o')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['Frozen + No Augmentation', 'Frozen + Augmentation', 'Last Conv Trainable + No Augmentation',
            'Last Conv Trainable + Augmentation', 'Fully Trainable + No Augmentation', 'Fully Trainable + Augmentation',
            'Randomly Initialized Encoder + Augmentation'], loc='lower right')
plt.grid()
plt.savefig('cifar10_bigan_aug_trainable_compar.png')


# Plot comparison graph
plt.figure()
plt.plot(num_unlabelled, classifier_pretrained_acc, '-o', num_unlabelled, classifier_random_acc, '-o')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['BiGAN Encoder', 'Randomly Initialized Encoder'], loc='lower right')
plt.grid()
plt.savefig('cifar10_pretrained_fully_sup_compar.png')