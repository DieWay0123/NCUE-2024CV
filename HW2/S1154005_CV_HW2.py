import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import enum
import queue

SOBEL_GX = np.array(
    [[-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]]
  , dtype=np.float32)
SOBEL_GY = np.array(
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
  , dtype=np.float32)
PREWITT_GX = np.array(
    [[-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]]
  , dtype=np.float32)
PREWITT_GY = np.array(
    [[-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]]
  , dtype=np.float32)

def normalize(image):
  min_val = np.min(image)
  max_val = np.max(image)
  #避免/0
  if min_val == max_val:
    return np.zeros(image.shape, dtype=np.uint8)
  normalized_image = (image - min_val)/(max_val-min_val)*255
  return normalized_image.astype(np.uint8)

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
  #image = image.astype(np.float32)
  pad_size = ksize//2
  padded_image = np.pad(image, pad_size, mode = 'reflect')
  gx = np.zeros(image.shape, dtype=np.float32)
  gy = np.zeros(image.shape, dtype=np.float32)

  for y in range(height):
    for x in range(width):
      gx[y, x] = np.sum(padded_image[y:y+ksize, x:x+ksize]*operator_x)
      gy[y, x] = np.sum(padded_image[y:y+ksize, x:x+ksize]*operator_y)
  return gx, gy

def sobel(image, method=1):
  new_image = np.zeros_like(image)
  gx, gy = cal_gradient(image, SOBEL_GX, SOBEL_GY)
  if method == 1:
    new_image = np.abs(gx)+np.abs(gy)
    new_image = normalize(new_image)
  else:
    new_image = np.sqrt(gx**2+gy**2)
    new_image = normalize(new_image)
  return new_image

def prewitt(image, method=1):
  new_image = np.zeros_like(image)
  gx, gy = cal_gradient(image, PREWITT_GX, PREWITT_GY)
  if method == 1:
    new_image = np.abs(gx)+np.abs(gy)
    new_image = normalize(new_image)
  else:
    new_image = np.sqrt(gx**2+gy**2)
    new_image = normalize(new_image)
  return new_image

def canny(image, TH=80, TL=20, ksize=3):
  gaussian_image = cv2.GaussianBlur(image, (3, 3), 0)
  Gx, Gy = cal_gradient(gaussian_image, SOBEL_GX, SOBEL_GY)
  G_magnitude = np.sqrt(Gx**2+Gy**2)
  #防止除0
  G_theta = np.rad2deg(np.arctan(Gy/(Gx+ 10e-9)))
  G_dir = np.zeros(shape=G_theta.shape, dtype=np.uint8)
  G_supressed = np.zeros(G_magnitude.shape, dtype=np.float32)

  H, W = image.shape

  class DIRECTION(enum.IntEnum):
    VERTICAL = 1, #67.5~112.5
    HORIZON = 2, #22.5~-22.5
    SLASH = 3, #112.5~157.5
    BACK_SLASH = 4 #22.5~67.5
    ZERO = 0

  for y in range(1, H-1):
    for x in range(1, W-1):
      pixel = G_theta[y,x]
      if((pixel < 67.5 and pixel >= 22.5) or (pixel >= -157.5 and pixel < -112.5)):
        G_dir[y, x] = DIRECTION.SLASH
      elif(pixel < 112.5 and pixel >= 67.5 or (pixel < -67.5 and pixel >= -112.5)):
        G_dir[y, x] = DIRECTION.VERTICAL
      elif(pixel < -22.5 and pixel >= -67.5 or (pixel < 157.5 and pixel >= 112.5)):
        G_dir[y, x] = DIRECTION.BACK_SLASH
      else:
        G_dir[y, x] = DIRECTION.HORIZON
  #做non-maxima supression
  for y in range(1, H-1):
    for x in range(1, W-1):
      cur_dir = G_dir[y, x]
      cur_magnitude = G_magnitude[y, x]
      v1 = 0
      v2 = 0 #鄰居
      if(cur_dir == DIRECTION.VERTICAL):
        v1 = G_magnitude[y+1, x]
        v2 = G_magnitude[y-1, x]
      elif(cur_dir == DIRECTION.HORIZON):
        v1 = G_magnitude[y, x+1]
        v2 = G_magnitude[y, x-1]
      elif(cur_dir == DIRECTION.SLASH):
        v1 = G_magnitude[y+1, x+1]
        v2 = G_magnitude[y-1, x-1]
      elif(cur_dir == DIRECTION.BACK_SLASH):
        v1 = G_magnitude[y-1, x+1]
        v2 = G_magnitude[y+1, x-1]
      else:
        v1 = 255
        v2 = 255
      
      if(cur_magnitude < v1 or cur_magnitude < v2):
        G_supressed[y, x] = 0
      else:
        G_supressed[y, x] = cur_magnitude
  #double threshold
  result_image = np.zeros(G_supressed.shape, dtype=np.uint8)
  G_strong_y, G_strong_x = np.where(G_supressed >=TH)
  G_weak_y, G_weak_x = np.where((G_supressed >= TL) & (G_supressed < TH))

  result_image[G_strong_y, G_strong_x] = 255
  result_image[G_weak_y, G_weak_x] = TL

  for y in range(1, H-1):
    for x in range(1, W-1):
      if result_image[y, x] == TL:
        flag = False
        for i in range(-1, 2):
          for j in range(-1, 2):
            if result_image[y+i, x+j] == 255:
              flag = True
              result_image[y][x] = 255
        if not flag:
          result_image[y][x] = 0
  return result_image


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
          if kernel[i][j] == image[y][x]:
            new_image[cur_y][cur_x] = kernel[i][j]

  return new_image

def erode(image, kernel):
  H, W = image.shape

  #一開始設定result_image全部為黑的
  #後續判斷時若"kernel有整個在圖內則kernel該點會是255(白的)，而其他部分被erode掉(黑的)"
  #new_image = np.full(image.shape, 0)
  new_image = np.full((image.shape), 0, dtype=np.uint8)
  pad_size = max(len(arr) for arr in kernel)
  pad_size = max(pad_size, kernel.shape[0])
  pad_size = pad_size//2

  pixels = []
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
        pixels.append((y,x))
  for y, x in pixels:
    new_image[y][x] = 255

  return new_image

def opening(image, kernel):
  return dilate(erode(image, kernel), kernel)

def closing(image, kernel):
  return erode(dilate(image, kernel), kernel)

if __name__ == "__main__":
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
  image_sobel = sobel(image)
  image_prewitt = prewitt(image)
  image_canny = canny(image)

  #kernel = np.zeros((5, 5), dtype=np.uint8)
  kernel = np.full((5,5), 255, dtype=np.uint8)
  binary_image = cv2.imread('binary.png', cv2.IMREAD_GRAYSCALE)
  image_dilate = dilate(binary_image, kernel)
  image_erode = erode(binary_image,  kernel)
  image_opening = opening(binary_image,  kernel)
  image_closing = closing(binary_image,  kernel)

  cv2.imwrite("sobel.png", image_sobel)
  cv2.imwrite("prewitt.png", image_prewitt)
  cv2.imwrite("canny.png", image_canny)

  cv2.imwrite("dilation.png", image_dilate)
  cv2.imwrite("erosion.png", image_erode)
  cv2.imwrite("opening.png", image_opening)
  cv2.imwrite("closing.png", image_closing)

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
  
  plt.subplot(1, 4, 4)
  plt.title("Handmade canny")
  plt.imshow(image_canny, cmap='gray')
  plt.show()

  # Morphology Part
  plt.figure()
  plt.subplot(1, 5, 1)
  plt.title("Original Image")
  plt.imshow(binary_image, cmap='gray')

  plt.subplot(1, 5, 2)
  plt.title("Dilation Image")
  plt.imshow(image_dilate, cmap='gray')

  plt.subplot(1, 5, 3)
  plt.title("Erosion Image")
  plt.imshow(image_erode, cmap='gray')
  
  plt.subplot(1, 5, 4)
  plt.title("Opening Image")
  plt.imshow(image_opening, cmap='gray')

  plt.subplot(1, 5, 5)
  plt.title("Closing Image")
  plt.imshow(image_closing, cmap='gray')
  plt.show()


  