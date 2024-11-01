import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy

def gaussian_filter(image, kernel_size=3, sigma=1):
  H, W, C = image.shape

  #padding image
  pad = kernel_size // 2
  padded_image = np.zeros((H+pad*2, W+pad*2, 3), dtype=np.float64)
  padded_image[pad: pad+H, pad:pad+W] = image.copy().astype(np.float64)
  
  #gen gaussian kernel
  kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
  for x in range(-pad, -pad+kernel_size):
    for y in range(-pad, -pad+kernel_size):
      kernel[x+pad, y+pad] = np.exp( -(x**2+y**2)/(2*(sigma**2)))
  kernel /= (2*np.pi*sigma*sigma)
  kernel /= np.sum(kernel)

  #filter the image
  new_image = padded_image.copy()
  for x in range(H):
    for y in range(W):
      for c in range(C):
        new_image[pad+x, pad+y, c] = np.sum(kernel*padded_image[x: x+kernel_size, y: y+kernel_size, c])
        #pad+0~pad+511
  
  new_image = new_image[pad: pad+H, pad: pad+W].astype(np.uint8)
  return new_image

def median_filter(image, kernel_size=3):
  H, W, C = image.shape[0], image.shape[1], image.shape[2]
  pad = kernel_size//2
  padded_image = np.zeros((H+pad*2, W+pad*2, 3), dtype=np.float64)
  padded_image[pad: pad+H, pad:pad+W] = image.copy().astype(np.float64)
  new_image = image.copy()

  #filtering
  for x in range(H):
    for y in range(W):
      for c in range(C): 
        kernel = padded_image[x: x+kernel_size, y:y+kernel_size, c]
        new_image[x, y, c] = np.median(kernel.reshape(-1))

  new_image = new_image[pad:pad+H, pad:pad+W].astype(np.uint8)
  return new_image

if __name__ == "__main__":
  image = cv2.imread("noise.bmp", cv2.IMREAD_UNCHANGED)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #vector_median_filter_image = vector_median_filter(image, kernel_size=5)
  gaussian_image = gaussian_filter(image, kernel_size=125, sigma=5)
  median_image = median_filter(image, kernel_size=5)

  plt.figure(1, figsize=(10, 5))
  plt.subplot(1, 3, 1)
  plt.title("Original Image")
  plt.imshow(image)
  
  plt.subplot(1, 3, 2)
  plt.title("Gasussian filter")
  plt.imshow(gaussian_image)

  plt.subplot(1, 3, 3)
  plt.title("Median filter")
  plt.imshow(median_image)
  plt.show()