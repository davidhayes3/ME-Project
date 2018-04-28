import numpy as np
import matplotlib.pyplot as plt


num_unlabelled = [200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]

pretrained_acc = np.loadtxt('Results/classifier1.txt', dtype=np.float32)
pretrained_aug_acc = np.loadtxt('Results/classifier2.txt', dtype=np.float32)
pretrained_lastconv_acc = np.loadtxt('Results/classifier3.txt', dtype=np.float32)
pretrained_lastconv_aug_acc = np.loadtxt('Results/classifier4.txt', dtype=np.float32)
random_acc = np.loadtxt('Results/classifier5.txt', dtype=np.float32)
random_aug_acc = np.loadtxt('Results/classifier6.txt', dtype=np.float32)

# =====================================
# Visualize results
# =====================================

# Plot comparison graph
plt.figure()
plt.plot(num_unlabelled, pretrained_acc, '-o', num_unlabelled, pretrained_aug_acc,
         num_unlabelled, pretrained_lastconv_acc, '-o', num_unlabelled, pretrained_lastconv_aug_acc, '-o',
         num_unlabelled, random_aug_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['BiGAN Encoder + No Augmentation', 'BiGAN Encoder + Augmentation',
            'BiGAN Encoder (Final Conv Layer Trainable) + No Augmentation', 'BiGAN Encoder (Final Conv Layer Trainable) + Augmentation',
            'Randomly Initialized Encoder + Augmentation'], loc='lower right')
plt.grid()
plt.savefig('cifar10_bigan_aug_compar.png')

# Plot comparison graph
plt.figure()
plt.plot(num_unlabelled, pretrained_acc, '-o', num_unlabelled, random_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['BiGAN Encoder', 'Randomly Initialized Encoder'], loc='lower right')
plt.grid()
plt.savefig('cifar10_pretrained_fully_sup_compar.png')


# Plot comparison graph
plt.figure()
plt.plot(num_unlabelled, pretrained_acc, '-o', num_unlabelled, pretrained_lastconv_acc, '-o', num_unlabelled, random_acc, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['BiGAN Encoder', 'BiGAN Encoder (Final Conv Layer Trainable)', 'Randomly Initialized Encoder'], loc='lower right')
plt.grid()
plt.savefig('cifar10_pretrained_fully_sup_compar_2.png')