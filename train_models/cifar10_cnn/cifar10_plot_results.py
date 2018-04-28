import numpy as np
import matplotlib.pyplot as plt


class1 = np.loadtxt('Results/classifier1.txt', dtype=np.float32)
class2 = np.loadtxt('Results/classifier2.txt', dtype=np.float32)
class3 = np.loadtxt('Results/classifier3.txt', dtype=np.float32)
class4 = np.loadtxt('Results/classifier4.txt', dtype=np.float32)
class5 = np.loadtxt('Results/classifier5.txt', dtype=np.float32)
class6 = np.loadtxt('Results/classifier6.txt', dtype=np.float32)

num_unlabelled = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]

# Plot comparison graph
plt.figure()

plt.plot(num_unlabelled, class1, '-o', num_unlabelled, class2, '-o', num_unlabelled, class3, '-o',
         num_unlabelled, class4, '-o', num_unlabelled, class5, '-o')
plt.title('Test Accuracy vs No. of Labelled Examples used for Training')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('No. of labelled examples')
plt.legend(['Basic AE Encoder', 'DAE Encoder', 'AAE Encoder', 'VAE Encoder',
            'BiGAN Encoder'], loc='lower right')
plt.grid()
plt.savefig('cifar10_model_compar.png')
plt.show()