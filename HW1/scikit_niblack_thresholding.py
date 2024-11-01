import numpy as np
from skimage.filters import threshold_niblack
from skimage.data import page
from matplotlib import pyplot as plt
import cv2

if __name__ == "__main__":
  raw_image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
  raw_image_ndarray = np.asarray(raw_image[:, :])
  #raw_image = page()
  niblack_threshold = threshold_niblack(raw_image, 25, 0.2)
  binary_niblack = raw_image_ndarray > niblack_threshold
  
  plt.figure(1, figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.title("org_image")
  plt.imshow(raw_image, cmap=plt.cm.gray)

  plt.subplot(1, 2, 2)
  plt.title("new_image")
  plt.imshow(binary_niblack, cmap=plt.cm.gray)
  plt.show()