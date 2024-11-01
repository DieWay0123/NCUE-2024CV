import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

def unnormalize(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    0-1 -> 0-255
    '''
    max_value = np.max(image)
    min_value = np.min(image)

    delta = max_value - min_value

    if delta == 0:
        new_image = np.zeros_like(image)
    else:
        new_image = ((image - min_value) / delta * 255).astype(np.uint8)

    return new_image

def normalize(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    0-255 -> 0-1
    '''
    min_value = np.min(image)
    max_value = np.max(image)

    delta = max_value - min_value

    if delta == 0:
        new_image = np.zeros_like(image)
    else:
        new_image = image.astype(np.float32) / delta

    return new_image

def sobel(image, method=0):
  height, width = image.shape
  new_image = np.zeros_like(image)
  normalizeed_image = normalize(image)
  
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
  
  for y in range(height-2):
    for x in range(width-2):
      gx = np.sum(np.multiply(GX, normalizeed_image[y:y+3, x:x+3]))
      gy = np.sum(np.multiply(GY, normalizeed_image[y:y+3, x:x+3]))
      if method == 1:
        new_image[y+1, x+1] = np.abs(gx)+np.abs(gy)
      else:
        new_image[y+1, x+1] = np.sqrt(gx**2+gy**2)
        
  return unnormalize(new_image)

if __name__ == "__main__":
  image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
  image_sobel = sobel(image)
  cv2Sobel = cv2.Sobel(image, ddepth=-1, dx=1, dy=0)
  
  cv2.imwrite('cv2Sobel.bmp', cv2Sobel)
  cv2.imwrite('customSobel.bmp', image_sobel)
  plt.figure()
  plt.imshow(image_sobel, cmap='gray')
  plt.show()