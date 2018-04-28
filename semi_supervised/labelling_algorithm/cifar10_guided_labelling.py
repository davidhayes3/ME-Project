from sklearn.metrics import confusion_matrix
from train_models.cifar10_cnn.cifar10_models import deterministic_encoder_model
from common_models.classifier_models import classifier_e_frozen_model, classifier_e_trainable_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functions.data_funcs import get_cifar10
import numpy as np
import keras
import itertools
import matplotlib.pyplot as plt


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

num_classes = 10
initial_num_labels = 1000
init_num_labels_per_class = int(initial_num_labels / num_classes)
pretrained_encoder_classifier_path = 'cifar10_pretrained_encoder_cnn.h5'
fully_supervised_classifier_path = 'cifar10_fully_supervised_cnn.h5'


# =====================================
# Load dataset
# =====================================

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = get_cifar10()

y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)


# =====================================
# Instantiate and compile models
# =====================================

# Instantiate and load pre-trained encoder
pretrained_encoder = deterministic_encoder_model()
pretrained_encoder.load_weights('cifar10_bigan_determ_encoder.h5')
pretrained_encoder.trainable = False

# Instanatiate random encoder
random_encoder = deterministic_encoder_model()


# Instantiate classifiers
pretrained_classifier = classifier_e_frozen_model(pretrained_encoder)
fully_supervised_classifier = classifier_e_trainable_model(random_encoder)


# Compile classifiers
pretrained_classifier.compile(loss='categorical_crossentropy',
                              optimizer='adadelta',
                              metrics=['accuracy'])

fully_supervised_classifier.compile(loss='categorical_crossentropy',
                                    optimizer='adadelta',
                                    metrics=['accuracy'])


# =====================================
# Set parameters for labeling algorithm
# =====================================

num_labels_added_per_iter = 1000

# Compute maximum number of iterations for algorithm (full training set is labelled)
max_num_iterations = np.int(X_train.shape[0] / (2 * num_labels_added_per_iter))

# Create arrays to hold number of labels
pretrained_num_labels = np.zeros(max_num_iterations)
fully_num_labels = np.zeros(max_num_iterations)

# Create arrays to record performacnce of classifiers
pretrained_acc = np.zeros(max_num_iterations, dtype=np.float32)
fully_supervised_acc = np.zeros(max_num_iterations, dtype=np.float32)

# Create vector with name of all classes
classes = np.arange(num_classes)


# =====================================
# Generate initial training set for classifier
# =====================================

# Get initial data examples to train on
indices_initial = np.empty(0)

# Modify training set to contain set number of labels for each class
for class_index in range(num_classes):
    # Generate training set with even class distribution over all labels
    indices = [i for i, y in enumerate(y_train) if y == classes[class_index]]
    indices = np.asarray(indices)
    indices = indices[0:init_num_labels_per_class]
    indices_initial = np.concatenate((indices_initial, indices))

# Sort indices so class examples are mixed up
indices_initial = np.sort(indices_initial)
indices_initial = indices_initial.astype(np.int)

# Reduce training vectors
x_train_initial = X_train[indices_initial]
y_train_initial = y_train[indices_initial]

# Convert label vectors to one-hot vectors
y_train_initial = keras.utils.to_categorical(y_train_initial, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)


# =====================================
# Train classifiers with initial dataset
# =====================================

# Set training hyper-parameters
epochs = 100
batch_size = 128
patience = 5
val_split = 1/10.

# Specify callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')


# ----------------------------
# Train classifier with pre-trained encoder
# ----------------------------

# Specify callbacks
model_checkpoint = ModelCheckpoint(pretrained_encoder_classifier_path, monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='min')
pretrained_callbacks = [early_stopping, model_checkpoint]

# Train classifier
pretrained_classifier.fit(x_train_initial, y_train_initial,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     callbacks=pretrained_callbacks,
                     shuffle=True,
                     validation_split=val_split)

# Load weights of best classifier
pretrained_classifier.load_weights(pretrained_encoder_classifier_path)

# Compute and print test accuracy of model
score = pretrained_classifier.evaluate(X_test, y_test_one_hot, verbose=0)
pretrained_acc[0] = 100 * score[1]
print('Pretrained Encoder Classifier: Overall test accuracy (%) with ' + str(init_num_labels_per_class) + ' labeled examples per class: '
              + str(pretrained_acc[0]))

pretrained_num_labels[0] = initial_num_labels


# ----------------------------
# Train fully supervised classifier
# ----------------------------

# Specify callbacks
model_checkpoint = ModelCheckpoint(fully_supervised_classifier_path, monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='min')
fully_supervised_callbacks = [early_stopping, model_checkpoint]

# Train classifier
fully_supervised_classifier.fit(x_train_initial, y_train_initial,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     callbacks=fully_supervised_callbacks,
                     shuffle=True,
                     validation_split=val_split)

# Load weights of best classifier
fully_supervised_classifier.load_weights(fully_supervised_classifier_path)

# Compute and print test accuracy of model
score = fully_supervised_classifier.evaluate(X_test, y_test_one_hot, verbose=0)
fully_supervised_acc[0] = 100 * score[1]
print('Fully Supervised Classifier: Overall test accuracy (%) with ' + str(init_num_labels_per_class) + ' labeled examples per class: '
              + str(fully_supervised_acc[0]))

fully_num_labels[0] = initial_num_labels


# =====================================
# Guided labelling for pretrained classifier
# =====================================

# Create unlabelled and labelled set
x_train_unlabelled = X_train
y_train_unlabelled = y_train
x_train_labelled = np.empty([0, 32, 32, 3])
y_train_labelled = np.empty([0, 1])


# Loop until test accuracy does not improve with additional examples
for iteration in range(1, max_num_iterations):

    print('\nIteration ' + str(iteration + 1) + '\n')

    if iteration == 2:
        y_train_labelled = y_train_labelled.reshape((y_train_labelled.shape[0],))


    # Calculate entropy of classifier for all examples in unlabelled set
    predictions = pretrained_classifier.predict(x_train_unlabelled)
    x_train_unlabelled_entropy = (-predictions * np.log2(predictions)).sum(axis=1)

    # Find indices of examples with 1000 highest entropy in unlabelled set
    max_entropy_indices = x_train_unlabelled_entropy.argsort()[-num_labels_added_per_iter:][::-1]


    # Add these examples to labelled set and remove from unlabelled set
    x_train_labelled = np.concatenate((x_train_labelled, x_train_unlabelled[max_entropy_indices]))
    y_train_labelled = np.concatenate((y_train_labelled, y_train_unlabelled[max_entropy_indices]))
    y_train_labelled_one_hot = keras.utils.to_categorical(y_train_labelled, num_classes)
    x_train_unlabelled = np.delete(x_train_unlabelled, max_entropy_indices, axis=0)
    y_train_unlabelled = np.delete(y_train_unlabelled, max_entropy_indices)


    # Train classifier
    print('Training on ' + str(len(x_train_labelled)) + ' newest most confusing examples\n')

    # Randomly initializae CNN
    pretrained_classifier = classifier_e_frozen_model(pretrained_encoder)

    # Train CNN
    pretrained_classifier.fit(x_train_labelled, y_train_labelled_one_hot,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         shuffle=True,
                         callbacks=pretrained_callbacks,
                         validation_split=val_split)

    pretrained_classifier.load_weights(pretrained_encoder_classifier_path)


    # Update and print test accuracy
    score = pretrained_classifier.evaluate(X_test, y_test_one_hot, verbose=0)
    pretrained_acc[iteration] = 100 * score[1]
    pretrained_num_labels[iteration] = x_train_labelled.shape[0]
    print('Test accuracy with ' + str(len(x_train_labelled)) + ' most confusing examples labelled: '
          + str(pretrained_acc[iteration]) + '%\n')


# =====================================
# Guided labelling for fully supervised classifier
# =====================================

# Create unlabelled and labelled set
x_train_unlabelled_new = X_train
y_train_unlabelled_new = y_train
x_train_labelled = np.empty([0, 32, 32, 3])
y_train_labelled = np.empty([0, 1])

# Loop until test accuracy is at state of art level
for iteration in range(1, max_num_iterations):

    print('\nIteration ' + str(iteration + 1) + '\n')

    if iteration == 2:
        y_train_labelled = y_train_labelled.reshape((y_train_labelled.shape[0],))

    # Calculate entropy of classifier for all examples in unlabelled set
    predictions = fully_supervised_classifier.predict(x_train_unlabelled_new)
    x_train_unlabelled_entropy = (-predictions * np.log2(predictions)).sum(axis=1)


    # Find indices of examples with 1000 highest entropy in unlabelled set
    max_entropy_indices = x_train_unlabelled_entropy.argsort()[-num_labels_added_per_iter:][::-1]


    # Add these examples to labelled set and remove from unlabelled set
    x_train_labelled = np.concatenate((x_train_labelled, x_train_unlabelled_new[max_entropy_indices]))
    y_train_labelled = np.concatenate((y_train_labelled, y_train_unlabelled_new[max_entropy_indices]))
    y_train_labelled_one_hot = keras.utils.to_categorical(y_train_labelled, num_classes)
    x_train_unlabelled_new = np.delete(x_train_unlabelled_new, max_entropy_indices, axis=0)
    y_train_unlabelled_new = np.delete(y_train_unlabelled_new, max_entropy_indices)


    # Train classifier
    print('Training on ' + str(len(x_train_labelled)) + ' most confusing examples\n')

    # Randomly initialize CNN
    random_encoder = deterministic_encoder_model()
    fully_supervised_classifier = classifier_e_trainable_model(random_encoder)

    # Train CNN
    fully_supervised_classifier.fit(x_train_labelled, y_train_labelled_one_hot,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            callbacks=fully_supervised_callbacks,
            validation_split=val_split)

    fully_supervised_classifier.load_weights(fully_supervised_classifier_path)


    # Update and print test accuracy
    score = fully_supervised_classifier.evaluate(X_test, y_test_one_hot, verbose=0)
    fully_supervised_acc[iteration] = 100 * score[1]
    fully_num_labels[iteration] = x_train_labelled.shape[0]
    print('Test accuracy with ' + str(len(x_train_labelled)) + ' most confusing examples labelled: '
          + str(fully_supervised_acc[iteration]) + '%\n')


# Print accuracies
print(pretrained_acc)
print(fully_supervised_acc)


# Save results to file
np.savetxt('classifier1_numlabels.txt', pretrained_num_labels, fmt='%d')
np.savetxt('classifier2_numlabels.txt', fully_num_labels, fmt='%d')
np.savetxt('classifier1_acc.txt', pretrained_acc, fmt='%f')
np.savetxt('classifier2_acc.txt', fully_supervised_acc, fmt='%f')


# =====================================
# Visualize results
# =====================================

plt.figure()

plt.plot(pretrained_num_labels, pretrained_acc)
plt.xlabel('No. of labelled examples available')
plt.ylabel('Test Accuracy (%)')
plt.savefig('pretrained_encoder_guided_labeling.png')


plt.figure()

plt.plot(fully_num_labels, fully_supervised_acc)
plt.xlabel('No. of labelled examples available')
plt.ylabel('Test Accuracy (%)')
plt.savefig('fully_supervised_guided_labeling.png')


'''# Generate confusion matrix

predictions = cnn.predict_classes(x_test)

plt.figure(2)

cm = confusion_matrix(y_test, predictions)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, num_classes)
plt.yticks(tick_marks, num_classes)

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.title('MNIST Confusion Matrix')
plt.show()'''