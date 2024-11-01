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
PREWITT_GX = np.array(
    [[-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]]
  , dtype=np.float32)
PREWITT_GY = np.array(
    [[-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]]
  , dtype=np.float32)

def gaussian_filter(image, kernel_size=3, sigma=1):
  H, W= image.shape

  #padding image
  pad = kernel_size // 2
  padded_image = np.zeros((H+pad*2, W+pad*2), dtype=np.float64)
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
      new_image[pad+x, pad+y] = np.sum(kernel*padded_image[x: x+kernel_size, y: y+kernel_size])
      #pad+0~pad+511
  
  new_image = new_image[pad: pad+H, pad: pad+W].astype(np.uint8)
  return new_image

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
        cur_y = y+i-1+pad_size
        for j in range(ksize):
            cur_x = x+j-1+pad_size
            gx_cur[i, j] = operator_x[i, j]*padded_image[cur_y, cur_x]
            gy_cur[i, j] = operator_y[i, j]*padded_image[cur_y, cur_x]
      gx[y, x] = np.sum(gx_cur)
      gy[y, x] = np.sum(gy_cur)

  return gx, gy

def sobel(image, method=1):
  new_image = np.zeros_like(image)
  gx, gy = cal_gradient(image, SOBEL_GX, SOBEL_GY)
  if method == 1:
    new_image = np.abs(gx)+np.abs(gy)
    new_image = (new_image+abs(new_image))/2 # val < 0 = 0
    new_image[new_image > 255] = 255 # val > 255 = 255
  else:
    new_image = np.sqrt(gx**2+gy**2)
    new_image = (new_image+abs(new_image))/2 # val < 0 = 0
    new_image[new_image > 255] = 255 # val > 255 = 255
  return new_image

def prewitt(image, method=1):
  new_image = np.zeros_like(image)
  gx, gy = cal_gradient(image, PREWITT_GX, PREWITT_GY)
  if method == 1:
    new_image = np.abs(gx)+np.abs(gy)
    new_image = (new_image+abs(new_image))/2 # val < 0 = 0
    new_image[new_image > 255] = 255 # val > 255 = 255
  else:
    new_image = np.sqrt(gx**2+gy**2)
    new_image = (new_image+abs(new_image))/2 # val < 0 = 0
    new_image[new_image > 255] = 255 # val > 255 = 255
  return new_image

def dilate(image, kernel):
  H, W = image.shape
  
  new_image = image.copy()
  pad_size = max(len(arr) for arr in kernel)
  pad_size = max(pad_size, kernel.shape[0])
  pad_size = pad_size//2
  
  for y in range(H):
    for x in range(W):
      for i in range(kernel.shape[0]):
        cur_y = y-i+pad_size
        for j in range(len(kernel[i])):
          cur_x = x-j+pad_size
          if cur_y < 0 or cur_y >= H or cur_x < 0 or cur_x >= W:
            continue
          if kernel[i][j] == 0 and image[y][x] == 0:
            new_image[cur_y][cur_x] = 0

  return new_image

def erode(image, kernel):
  H, W = image.shape

  #一開始設定全部為白的
  #後續判斷時若"kernel有整個在圖內則kernel該點會是0(黑的)，且其他部分被erode掉(原本就是白的)"
  new_image = np.full(image.shape, 255)

  #new_image = np.copy(image)
  pad_size = max(len(arr) for arr in kernel)
  pad_size = max(pad_size, kernel.shape[0])
  pad_size = pad_size//2
  for y in range(H):
    for x in range(W):
      flag = True
      for i in range(kernel.shape[0]):
        cur_y = y+i-pad_size
        for j in range(kernel[i].size):
          cur_x = x+j-pad_size
          if cur_y < 0 or cur_y >= H or cur_x < 0 or cur_x >= W:
            continue
          if image[cur_y][cur_x] != kernel[i][j]:
            flag = False
            break
      if flag == True:
        new_image[y][x] = 0


  return new_image

def opening(image, kernel):
  return dilate(erode(image, kernel), kernel)

def closing(image, kernel):
  return erode(dilate(image, kernel), kernel)

if __name__ == "__main__":
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
  gaussian_image = gaussian_filter(image)
  image_sobel = sobel(gaussian_image)
  image_prewitt = prewitt(gaussian_image)

  kernel = np.zeros((5, 5), dtype=np.uint8)
  binary_image = cv2.imread('binary.png', cv2.IMREAD_GRAYSCALE)
  image_dilate = dilate(binary_image, kernel)
  image_erode = erode(binary_image,  kernel)
  image_opening = opening(binary_image,  kernel)
  image_closing = closing(binary_image,  kernel)

  # Edge Detection Part
  plt.figure()
  plt.subplot(1, 4, 1)
  plt.title("Original Image")
  plt.imshow(image, cmap='gray')

  plt.subplot(1, 4, 2)
  plt.title("Handmade Sobel")
  plt.imshow(image_sobel, cmap='gray')

  plt.subplot(1, 4, 3)
  plt.title("Handmade Prewitt")
  plt.imshow(image_prewitt, cmap='gray')
  plt.show()

  # Morphology Part
  plt.figure()
  plt.subplot(1, 5, 1)
  plt.title("Original Image")
  plt.imshow(binary_image, cmap='gray')

  plt.subplot(1, 5, 2)
  plt.title("Dilate Image")
  plt.imshow(image_dilate, cmap='gray')

  plt.subplot(1, 5, 3)
  plt.title("Erode Image")
  plt.imshow(image_erode, cmap='gray')
  
  plt.subplot(1, 5, 4)
  plt.title("Opening Image")
  plt.imshow(image_opening, cmap='gray')

  plt.subplot(1, 5, 5)
  plt.title("Closing Image")
  plt.imshow(image_closing, cmap='gray')
  plt.show()
  

  