import numpy as np
import matplotlib.pyplot as plt
import cv2

noise = np.random.normal(0, 1, (100, 100, 3))



fig, axs = plt.subplots(5, 5)
count = 0
for i in range(5):
	for j in range(5):
		axs[i, j].imshow(noise)
		axs[i, j].axis('off')

fig.savefig('noise_plt.png')
plt.close()