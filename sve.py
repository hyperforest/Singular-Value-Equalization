# import library
import numpy as np
import cv2

# load the image
image_name = 'tree.bmp'
image = cv2.imread(image_name, 0)

# create Gaussian matrix
G = np.random.normal(0.5, 1, image.shape)

# do Singular Value Decomposition
UX, SX, VXT = np.linalg.svd(image)
UG, SG, VGT = np.linalg.svd(G)

# calculate constant E and the equalized image
E = max(SG) / max(SX)
output = (E * image * 255).astype('uint8')

# save output image
cv2.imwrite('tree_sve.bmp', output)
