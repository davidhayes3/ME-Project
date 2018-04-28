import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from mnist_mlp_models import encoder_model, vae_encoder_model
from common_models.common_models import vae_encoder_sampling_model
from functions.data_funcs import get_mnist
from common_models.classifier_models import mnist_classifier_e_frozen_model, mnist_classifier_e_trainable_model


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


# =====================================
# Load data
# =====================================

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = get_mnist()

# Distinguish training sets for models
(x_train_ae, _), (x_test_ae, _) = get_mnist()
(x_train_gan, _), (x_test_gan, _) = get_mnist(gan=True)


# Label data is same for both
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)


# =====================================
# Instantiate and load models
# =====================================

# Instantiate encoders
basic_ae = encoder_model()
dae = encoder_model()
sae = encoder_model()
ce = encoder_model()
aae = encoder_model()
lr = encoder_model()
jlr = encoder_model()
bigan = encoder_model()
mod_bigan = encoder_model()
cnn = encoder_model()

vae_encoder = vae_encoder_model()
vae = vae_encoder_sampling_model(vae_encoder, latent_dim, img_shape, epsilon_std=0.05)


# Load pre-trained weights
model_path = 'Models/mnist'

basic_ae.load_weights(model_path + '_basic_ae_encoder.h5')
dae.load_weights(model_path + '_dae_encoder.h5')
sae.load_weights(model_path + '_sae_encoder.h5')
ce.load_weights(model_path + '_ce_encoder.h5')
aae.load_weights(model_path + '_aae_encoder.h5')
vae.load_weights(model_path + '_vae_encoder.h5')
lr.load_weights(model_path + '_lr_encoder.h5')
jlr.load_weights(model_path + '_jlr_encoder.h5')
bigan.load_weights(model_path + '_bigan_encoder.h5')
mod_bigan.load_weights(model_path + '_posthoc_bigan_encoder.h5')


# Freeze the parameters of all encoders
basic_ae.trainable = False
dae.trainable = False
sae.trainable = False
ce.trainable = False
aae.trainable = False
vae.trainable = False
lr.trainable = False
jlr.trainable = False
bigan.trainable = False
mod_bigan.trainable = False


# =====================================
# Train models
# =====================================

# Set training hyper-parameters
epochs = 100
batch_size = 128
val_split = 1/5.
patience = 10

# Specify training stop criterion and when to save model weights
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')


# Number of labelled examples to investigate
num_unlabelled = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 60000]
num_iterations = 5


# Arrays to hold accuracy of classifiers
classifier1_acc = np.zeros(len(num_unlabelled))
classifier2_acc = np.zeros(len(num_unlabelled))
classifier3_acc = np.zeros(len(num_unlabelled))
classifier4_acc = np.zeros(len(num_unlabelled))
classifier5_acc = np.zeros(len(num_unlabelled))
classifier6_acc = np.zeros(len(num_unlabelled))
classifier7_acc = np.zeros(len(num_unlabelled))
classifier8_acc = np.zeros(len(num_unlabelled))
classifier9_acc = np.zeros(len(num_unlabelled))
classifier10_acc = np.zeros(len(num_unlabelled))
classifier11_acc = np.zeros(len(num_unlabelled))



# Loop through each quantity of enquiry
for index, num in enumerate(num_unlabelled):

    classifier1_score = 0
    classifier2_score = 0
    classifier3_score = 0
    classifier4_score = 0
    classifier5_score = 0
    classifier6_score = 0
    classifier7_score = 0
    classifier8_score = 0
    classifier9_score = 0
    classifier10_score = 0
    classifier11_score = 0

    # Reduce size of training sets
    reduced_x_train_ae = x_train_ae[0:num, :, :, :]
    reduced_x_train_gan = x_train_gan[0:num, :, :, :]
    reduced_y_train = y_train_one_hot[0:num, :]

    # Average classification accuracy over num_iterations readings
    for iteration in range(num_iterations):

        # Print details of no. of labelled examples and iteration number
        print('Labelled Examples: ' + str(num) + ', Iteration: ' + str(iteration+1) + '/' + str(num_iterations))

        # Instantiate classfiers to be trained
        classifier1 = mnist_classifier_e_frozen_model(basic_ae)
        classifier2 = mnist_classifier_e_frozen_model(dae)
        classifier3 = mnist_classifier_e_frozen_model(sae)
        classifier4 = mnist_classifier_e_frozen_model(ce)
        classifier5 = mnist_classifier_e_frozen_model(vae)
        classifier6 = mnist_classifier_e_frozen_model(aae)
        classifier7 = mnist_classifier_e_frozen_model(lr)
        classifier8 = mnist_classifier_e_frozen_model(jlr)
        classifier9 = mnist_classifier_e_frozen_model(bigan)
        classifier10 = mnist_classifier_e_frozen_model(mod_bigan)
        classifier11 = mnist_classifier_e_trainable_model(cnn)


        # Compile models
        classifiers = (classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, classifier7, 
                       classifier8, classifier9, classifier10, classifier11)

        for classifier in classifiers:
            classifier.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])


        # =====================================
        # Train models
        # =====================================

        # Classifier 1
        model_checkpoint = ModelCheckpoint('Models/classifier_1.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier1.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier1.load_weights('Models/classifier_1.h5')
        score = classifier1.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier1_score += score[1]


        # Classifier 2
        model_checkpoint = ModelCheckpoint('Models/classifier_2.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier2.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier2.load_weights('Models/classifier_2.h5')
        score = classifier2.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier2_score += score[1]


        # Classifier 3
        model_checkpoint = ModelCheckpoint('Models/classifier_3.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier3.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier3.load_weights('Models/classifier_3.h5')
        score = classifier3.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier3_score += score[1]


        # Classifier 4
        model_checkpoint = ModelCheckpoint('Models/classifier_4.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier4.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier4.load_weights('Models/classifier_4.h5')
        score = classifier4.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier4_score += score[1]


        # Classifier 5
        model_checkpoint = ModelCheckpoint('Models/classifier_5.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier5.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier5.load_weights('Models/classifier_5.h5')
        score = classifier5.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier5_score += score[1]


        # Classifier 6
        model_checkpoint = ModelCheckpoint('Models/classifier_6.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier6.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier6.load_weights('Models/classifier_6.h5')
        score = classifier6.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier6_score += score[1]


        # Classifier 7
        model_checkpoint = ModelCheckpoint('Models/classifier_7.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier7.fit(reduced_x_train_gan, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier7.load_weights('Models/classifier_7.h5')
        score = classifier7.evaluate(x_test_gan, y_test_one_hot, verbose=0)
        classifier7_score += score[1]


        # Classifier 8
        model_checkpoint = ModelCheckpoint('Models/classifier_8.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier8.fit(reduced_x_train_gan, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier8.load_weights('Models/classifier_8.h5')
        score = classifier8.evaluate(x_test_gan, y_test_one_hot, verbose=0)
        classifier8_score += score[1]


        # Classifier 9
        model_checkpoint = ModelCheckpoint('Models/classifier_9.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier9.fit(reduced_x_train_gan, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier9.load_weights('Models/classifier_9.h5')
        score = classifier9.evaluate(x_test_gan, y_test_one_hot, verbose=0)
        classifier9_score += score[1]


        # Classifier 10
        model_checkpoint = ModelCheckpoint('Models/classifier_10.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier10.fit(reduced_x_train_gan, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier10.load_weights('Models/classifier_10.h5')
        score = classifier10.evaluate(x_test_gan, y_test_one_hot, verbose=0)
        classifier10_score += score[1]


        # Classifier 11
        model_checkpoint = ModelCheckpoint('Models/classifier_11.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier11.fit(reduced_x_train_ae, reduced_y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         shuffle=True,
                         callbacks=callbacks,
                         validation_split=val_split)

        classifier11.load_weights('Models/classifier_11.h5')
        score = classifier11.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier11_score += score[1]


    # Record average classification accuracy for each no. of labelled examples
    classifier1_acc[index] = 100 * classifier1_score / num_iterations
    classifier2_acc[index] = 100 * classifier2_score / num_iterations
    classifier3_acc[index] = 100 * classifier3_score / num_iterations
    classifier4_acc[index] = 100 * classifier4_score / num_iterations
    classifier5_acc[index] = 100 * classifier5_score / num_iterations
    classifier6_acc[index] = 100 * classifier6_score / num_iterations
    classifier7_acc[index] = 100 * classifier7_score / num_iterations
    classifier8_acc[index] = 100 * classifier8_score / num_iterations
    classifier9_acc[index] = 100 * classifier9_score / num_iterations
    classifier10_acc[index] = 100 * classifier10_score / num_iterations
    classifier11_acc[index] = 100 * classifier11_score / num_iterations


    # Save accuracies to file
    np.savetxt('Results/classifier1.txt', classifier1_acc, fmt='%f')
    np.savetxt('Results/classifier2.txt', classifier2_acc, fmt='%f')
    np.savetxt('Results/classifier3.txt', classifier3_acc, fmt='%f')
    np.savetxt('Results/classifier4.txt', classifier4_acc, fmt='%f')
    np.savetxt('Results/classifier5.txt', classifier5_acc, fmt='%f')
    np.savetxt('Results/classifier6.txt', classifier6_acc, fmt='%f')
    np.savetxt('Results/classifier7.txt', classifier7_acc, fmt='%f')
    np.savetxt('Results/classifier8.txt', classifier8_acc, fmt='%f')
    np.savetxt('Results/classifier9.txt', classifier9_acc, fmt='%f')
    np.savetxt('Results/classifier10.txt', classifier10_acc, fmt='%f')
    np.savetxt('Results/classifier11.txt', classifier11_acc, fmt='%f')


# Print accuracies
print(classifier1_acc)
print(classifier2_acc)
print(classifier3_acc)
print(classifier4_acc)
print(classifier5_acc)
print(classifier6_acc)
print(classifier7_acc)
print(classifier8_acc)
print(classifier9_acc)
print(classifier10_acc)
print(classifier11_acc)