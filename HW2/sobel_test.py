import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

SOBEL_GX = np.array(
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
  , dtype=np.float32)
SOBEL_GY = np.array(
    [[-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]]
  , dtype=np.float32)

def cal_gradient(image, operator_x, operator_y):
  height, width = image.shape
  ksize = 3

  pad_size = ksize//2
  padded_image = np.pad(image, pad_size)
  gx = np.zeros(image.shape, dtype=np.float32)
  gy = np.zeros(image.shape, dtype=np.float32)
  gx_cur = np.zeros((ksize, ksize), dtype=np.float32)
  gy_cur = np.zeros((ksize, ksize), dtype=np.float32)

  for y in range(height):
    for x in range(width):
      for i in range(ksize):
        cur_y = y+i-1
        for j in range(ksize):
            cur_x = x+j-1
            gx_cur[i, j] = operator_x[i, j]*padded_image[cur_y, cur_x]
            gy_cur[i, j] = operator_y[i, j]*padded_image[cur_y, cur_x]
      gx[y, x] = np.sum(gx_cur)
      gy[y, x] = np.sum(gy_cur)

  return gx, gy

def sobel(image, method=0):
  new_image = np.zeros_like(image)
  gx, gy = cal_gradient(image, SOBEL_GX, SOBEL_GY)
  '''   
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
  '''
  if method == 1:
    new_image = np.abs(gx)+np.abs(gy)
  else:
    new_image = np.sqrt(gx**2+gy**2)
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