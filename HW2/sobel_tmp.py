import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

def sobel(image, method=0):
  height, width = image.shape
  new_image = np.zeros_like(image)
  ksize=3
  
  GX = np.array(
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
  , dtype=np.float32)
  GY = np.array(
    [[-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]]
  , dtype=np.float32)
  
  gx = np.zeros((ksize, ksize), dtype=np.float32)
  gy = np.zeros((ksize, ksize), dtype=np.float32)

  for y in range(height):
    for x in range(width):
      for i in range(ksize):
        cur_y = y+i-1
        for j in range(ksize):
          cur_x = x+j-1

          if(cur_y < 0 or cur_y >=height or cur_x < 0 or cur_x >= width):
            gx[i][j] = image[i][j]
            gy[i][j] = image[i][j]            
          else:
            gx[i][j] = GX[i][j]*image[cur_y][cur_x]
            gy[i][j] = GY[i][j]*image[cur_y][cur_x]

      #gx = np.sum(np.multiply(GX, image[y:y+3, x:x+3]))
      #gy = np.sum(np.multiply(GY, image[y:y+3, x:x+3]))
      a = np.sum(gx)
      b = np.sum(gy)
      if method == 1:
        new_image[y, x] = np.abs(a)+np.abs(b)
      else:
        new_image[y, x] = np.sqrt(a**2+b**2)
  print(new_image)
  return new_image

if __name__ == "__main__":
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
  image_sobel = sobel(image)
  cv2Sobel = cv2.Sobel(image, ddepth=-1, dx=1, dy=0)
  
  cv2.imwrite('cv2Sobel.bmp', cv2Sobel)
  cv2.imwrite('customSobel.bmp', image_sobel)
  plt.figure()
  plt.imshow(image_sobel, cmap='gray')
  plt.show()