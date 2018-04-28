from __future__ import print_function, division
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from matplotlib import cm


# Set random seed for reproducibility
np.random.seed(12345)


# =====================================
# Define constants
# =====================================

img_dim = 4
img_rows = 2
img_cols = 2
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 2
num_classes = 16
image_path = 'Images/sd_pca'


# =====================================
# Load dataset
# =====================================

# Load dataset
X_train = np.loadtxt('Dataset/synthetic_dataset_x_train.txt', dtype=np.float32)
X_test = np.loadtxt('Dataset/synthetic_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/synthetic_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/synthetic_dataset_y_test.txt', dtype=np.int)


# =====================================
# Perform PCA Algorithm
# =====================================

pca = decomposition.PCA(n_components=latent_dim)
z = pca.fit_transform(X_train)


# =====================================
# Save 2D latent visualization
# =====================================

fig = plt.figure()
ax = fig.add_subplot(111)
colors = cm.Spectral(np.linspace(0, 1, num_classes))

xx = z[:, 0]
yy = z[:, 1]

# Plot 2D data points
for i in range(num_classes):
    ax.scatter(xx[y_train == i], yy[y_train== i], color=colors[i], label=i, s=5)

plt.axis('tight')

plt.savefig(image_path + '_latent_vis.png')
plt.close()