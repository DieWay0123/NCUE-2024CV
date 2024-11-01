import cv2
import numpy as np
from matplotlib import pyplot as plt

def sobel(image, method=1):
  height, width = image.shape
  new_image = np.zeros_like(image)
  
  GX = np.array(
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  , dtype=np.float32)
  GY = np.array(
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
  , dtype=np.float32)
  
  for y in range(height-2):
    for x in range(width-2):
      gx = np.sum(np.multiply(GX, image[y:y+3, x:x+3]))
      gy = np.sum(np.multiply(GY, image[y:y+3, x:x+3]))
      if method == 1:
        new_image[y+1, x+1] = np.abs(gx)+np.abs(gy)
      else:
        new_image[y+1, x+1] = np.sqrt(gx**2+gy**2)

  return new_image

if __name__ == 'main':
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
  
  image_sobel = sobel(image)
  cv2.imwrite('test.bmp', image_sobel)
  plt.figure()
  plt.imshow(image_sobel, cmap='gray')
  plt.show()