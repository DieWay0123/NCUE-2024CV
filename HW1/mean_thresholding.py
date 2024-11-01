import numpy as np
import cv2

def mean_thresholding(image, block_size, c):
  new_image = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      # calculate neighbor block
      x_min = max(x-block_size//2,0)
      x_max = min(x+block_size//2, image.shape[0]-1)

      y_min = max(y-block_size//2, 0)
      y_max = min(y+block_size//2, image.shape[1]-1)

      #calculate neighbor block mean as thershold for (x,y)
      local_mean_threshold = np.mean(image[x_min:x_max, y_min:y_max])

      if image[x,y] > local_mean_threshold-c:
        new_image[x,y] = 255
      else:
        new_image[x,y] = 0
      #print(f"{x_min} {x_max}")
  cv2.imwrite("mean_thresholding.png", new_image)
if __name__ == "__main__":
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
  mean_thresholding(image, block_size=15, c=2)