import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib inline

img_filenames = sorted(os.listdir('../../Datasets/images'))
imgs = [mpimg.imread(os.path.join('../../Datasets/images', img_filename)) for img_filen


fig, axes = plt.subplots(2, 2)
fig.figsize = (6, 6)
fig.dpi = 150
axes = axes.ravel()

labels = ['coast', 'beach', 'building', 'city at night']

for i in range(len(imgs)):
    axes[i].imshow(imgs[i])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_xlabel(labels[i])
plt.show()
