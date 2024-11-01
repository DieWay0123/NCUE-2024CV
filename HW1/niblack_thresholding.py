import numpy as np
from skimage.filters import threshold_niblack
from skimage.data import page
from matplotlib import pyplot as plt
import cv2

def niblack_thresholding(image, block_size, k):
  new_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      # calculate neighbor block
      x_min = max(x-block_size//2,0)
      x_max = min(x+block_size//2, image.shape[0]-1)

      y_min = max(y-block_size//2, 0)
      y_max = min(y+block_size//2, image.shape[1]-1)

      #calculate neighbor block mean as thershold for (x,y)
      local_mean = np.mean(image[x_min:x_max, y_min:y_max])
      local_standard_deviation = np.std(image[x_min:x_max, y_min:y_max].flatten(), ddof=0)
      local_threshold = local_mean + k*local_standard_deviation
      if image[x,y] > local_threshold:
        new_image[x,y] = 255
      else:
        new_image[x,y] = 0
  cv2.imwrite("niblack_thresholding.png", new_image)

if __name__ == "__main__":
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
  niblack_thresholding(image, block_size=11, k=0.2)