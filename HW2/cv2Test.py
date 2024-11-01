import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

  #sobel
  img_sobelx = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=3)
  img_sobely = cv2.Sobel(image,cv2.CV_8U,0,1,ksize=3)
  img_sobel = img_sobelx + img_sobely

  #prewitt
  kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
  kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
  img_prewittx = cv2.filter2D(image, -1, kernelx) #gaussian
  img_prewitty = cv2.filter2D(image, -1, kernely)
  img_prewitt = np.abs(img_prewittx)+np.abs(img_prewitty)

  print
  plt.figure()
  plt.title("sobel")
  plt.imshow(img_sobel, cmap='gray')
  plt.figure()
  plt.subplot(1, 3, 1)
  plt.imshow(img_prewittx, cmap='gray')
  plt.subplot(1, 3, 2)
  plt.imshow(img_prewitty, cmap='gray')
  plt.subplot(1, 3, 3)
  plt.title("prewitt")
  plt.imshow(img_prewitt, cmap='gray')
  plt.show()