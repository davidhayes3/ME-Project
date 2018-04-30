import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import matplotlib.gridspec as gridspec


# Function to plot training loss curves
def plot_train_accuracy(path, history):
    plt.figure()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.savefig(path + '_train_acc.png')
    plt.show()

# Function to plot training accuracy curves
def plot_train_loss(path, history):
    plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.savefig(path + '_train_loss.png')
    plt.show()


# Function to plot batch loss curves for generator and discriminator loss
def plot_gan_batch_loss(path, epochs, num_batches, d_batch_loss_trajectory, g_batch_loss_trajectory):
    plt.figure()

    batch_numbers = np.arange((epochs * num_batches)) + 1

    plt.plot(batch_numbers, d_batch_loss_trajectory, 'b-', batch_numbers, g_batch_loss_trajectory, 'r-')
    plt.legend(['Discriminator', 'Generator'], loc='upper right')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')

    plt.savefig(path + '_batchloss.png')
    plt.show()


# Function to plot epoch loss curves for g and d
def plot_gan_epoch_loss(path, epochs, d_epoch_loss_trajectory, g_epoch_loss_trajectory):
    plt.figure()

    epoch_numbers = np.arange(epochs) + 1

    plt.plot(epoch_numbers, d_epoch_loss_trajectory, 'b-', epoch_numbers, g_epoch_loss_trajectory, 'r-')
    plt.legend(['Discriminator', 'Generator'], loc='upper right')
    plt.xlabel('Epoch Number')
    plt.ylabel('Average Minibatch Loss')

    plt.savefig(path + '_epochloss.png')


# Function to plot discriminator accuracy over epochs
def plot_discriminator_acc(path, epochs, d_acc_trajectory):
    plt.figure()

    epoch_numbers = np.arange(epochs) + 1

    plt.plot(epoch_numbers, d_acc_trajectory)
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')

    plt.savefig(path + '_discriminator_acc.png')



# Function to plot reconstructions of test set examples
def save_reconstructions(path, num_classes, test_data, test_labels, generator, encoder, img_rows, img_cols, channels,
                         color=True, num_recons_per_class=10):
    # Get initial data examples to train on
    classes = np.arange(num_classes)
    test_digit_indices = np.empty(0)

    # Modify training set to contain set number of labels for each class
    for class_index in range(num_classes):
        # Generate training set with even class distribution over all labels
        indices = [i for i, y in enumerate(test_labels) if y == classes[class_index]]
        indices = np.asarray(indices)
        indices = indices[0:num_recons_per_class]
        test_digit_indices = np.concatenate((test_digit_indices, indices))

    test_digit_indices = test_digit_indices.astype(np.int)

    # Generate test and reconstructed digit arrays
    X_test = test_data[test_digit_indices]
    recon_x = generator.predict(encoder.predict(X_test))

    num_rows = num_classes
    num_cols = num_recons_per_class

    plt.figure(figsize=(num_rows, num_cols))

    gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=num_recons_per_class*[1],
                           wspace=0., hspace=0., top=0.8, bottom=0.2, left=0.2, right=0.8)

    for i in range(num_rows):
        for j in range(num_cols):
            if color is True:
                im = recon_x[i * num_cols + j].reshape(img_rows, img_cols, channels)
            if color is False:
                im = recon_x[i * num_cols + j].reshape(img_rows, img_cols)
                plt.gray()
            ax = plt.subplot(gs[i, j])
            plt.imshow(im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig(path + '_recons.png')



# Function to save images
def save_imgs(path, gen_imgs, epoch, img_rows, img_cols, channels, color=True):
    r, c = 5, 5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            if color is True:
                axs[i, j].imshow(gen_imgs[count].reshape(img_rows, img_cols, channels))
            elif color is False:
                axs[i, j].imshow(gen_imgs[count].reshape(img_rows, img_cols), cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig(path + '_gen_%d.png' % (epoch))
    plt.close()


# Function to plot 2D latent space visualizations
def save_latent_vis(path, data, labels, encoder, num_classes, epoch=None):

    z = encoder.predict(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))

    xx = z[:,0]
    yy = z[:,1]

    # Plot 2D data points
    for i in range(num_classes):
        ax.scatter(xx[labels == i], yy[labels == i], color=colors[i], label=i, s=5)

    plt.axis('tight')

    if epoch is None:
        plt.savefig(path + '_latent_vis.png')
    elif epoch is not None:
        plt.savefig(path + '_latent_vis_%d.png' % (epoch + 1))

    plt.close()