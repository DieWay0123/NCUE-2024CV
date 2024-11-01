import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def mean_thresholding(image, block_size=3, c=2):
  new_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      # calculate neighbor block
      x_min = max(x-block_size//2,0)
      x_max = min(x+block_size//2+1, image.shape[0]-1)
      y_min = max(y-block_size//2, 0)
      y_max = min(y+block_size//2+1, image.shape[1]-1)

      #calculate neighbor block mean as thershold for (x,y)
      local_mean_threshold = np.mean(image[x_min:x_max, y_min:y_max])

      #thresholding
      if image[x,y] > local_mean_threshold-c:
        new_image[x,y] = 255
      else:
        new_image[x,y] = 0
  return new_image

def niblack_thresholding(image, block_size=3, k=0.2):
  new_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      # calculate neighbor block
      x_min = max(x-block_size//2,0)
      x_max = min(x+block_size//2+1, image.shape[0]-1)
      y_min = max(y-block_size//2, 0)
      y_max = min(y+block_size//2+1, image.shape[1]-1)

      #calculate neighbor block standard_deviation as thershold for (x,y)
      local_mean = np.mean(image[x_min:x_max, y_min:y_max])
      local_standard_deviation = np.std(image[x_min:x_max, y_min:y_max].flatten(), ddof=0)
      local_threshold = local_mean + k*local_standard_deviation

      #thresholding
      if image[x,y] > local_threshold:
        new_image[x,y] = 255
      else:
        new_image[x,y] = 0
  return new_image

def otsu_thresholding(image, histogram):
  new_image = image.copy()
  max_variance_between, max_threshold = -999, 1
  for k in range(1, 256):
    #計算P(k)->前景, 背景的機率總和分別為w1,w2
    w1 = np.sum(histogram[:k])
    w2 = np.sum(histogram[k:])
    
    if w1 == 0 or w2 == 0:
      continue
    #計算mean intensity
    mean1 = np.sum(np.arange(0, k)*histogram[:k])/w1
    mean2 = np.sum(np.arange(k, 256)*histogram[k:])/w2
    meanG = np.sum(np.arange(0, 256)*histogram[:])/(w1+w2)

    variance_between = w1*(mean1-meanG)**2+w2*(mean2-meanG)**2

    #找k=1~255間最大的組間變數
    threshold_lst = []
    if variance_between > max_variance_between:
      threshold_lst.clear()
      threshold_lst.append(k)
      max_variance_between = variance_between
      max_threshold = np.mean(threshold_lst)
    elif variance_between == max_variance_between:
      threshold_lst.append(k)
      max_threshold = np.mean(threshold_lst)

  #thresholding
  for x in range(new_image.shape[0]):
    for y in range(new_image.shape[1]):
      if image[x, y] > max_threshold:
        new_image[x, y] = 255
      else:
        new_image[x, y] = 0
  return new_image

def Entropy(x):
  tmp = np.multiply(x, np.log(x+1e-5))
  tmp[np.isnan(tmp)] = 0
  return tmp

def entropy_thresholding(image, histogram):
  H = np.zeros(256)
  for k in range(1,256):
    w1 = np.sum(histogram[:k])
    w2 = np.sum(histogram[k:])

    if w1==0 or w2==0:
      continue
    
    #計算前景,背景熵
    h1 = -np.sum( Entropy(histogram[:k]/w1))
    h2 = -np.sum( Entropy(histogram[k:]/w2))
    entropy = h1+h2
    H[k] = entropy

  #thresholding
  threshold = np.argmax(H)
  new_image = image.copy()
  return new_image > threshold

if __name__ == "__main__":
  np.seterr(invalid='ignore')
  image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
  Mean_thresholding_image = mean_thresholding(image, 15, 2)
  Niblack_thresholding_image = niblack_thresholding(image, 15, -0.2)

  pixel_values = image.flatten()

  #calculate probability of image intensity from 0-255
  histogram, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256), density=True)
  otsu_image = otsu_thresholding(image, histogram)

  entropy_image = entropy_thresholding(image, histogram)

  plt.figure(1, figsize=(10, 5))
  plt.subplot(2, 3, 1)
  plt.title("Original Image")
  plt.imshow(image, cmap=plt.cm.gray)

  plt.subplot(2, 3, 2)
  plt.title("Mean thresholding(local)")
  plt.imshow(Mean_thresholding_image, cmap=plt.cm.gray)

  plt.subplot(2, 3, 3)
  plt.title("Niblack thresholding(local)")
  plt.imshow(Niblack_thresholding_image, cmap=plt.cm.gray)

  plt.subplot(2, 3, 5)
  plt.title("Otsu thresholding(global)")
  plt.imshow(otsu_image, cmap=plt.cm.gray)

  plt.subplot(2, 3, 6)
  plt.title("Entropy thresholding(global)")
  plt.imshow(entropy_image, cmap=plt.cm.gray)

  #plt.figure(2)
  #plt.bar(np.arange(0, 256), histogram, color='b', align='center')

  plt.show()