import cv2
import matplotlib
import numpy as np
import enum
import time
import math

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches

SOBEL_GX = np.array(
  [[-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]]
)
SOBEL_GY = np.array(
  [[-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]]  
)

PICTURE_ROOT_PATH = './pic'
RESULT_ROOT_PATH = './result'

def cal_gradient(image, operator_x, operator_y, ksize=3):
  H, W = image.shape
  image = image.astype(np.float32)
  pad = ksize // 2
  padded_image = np.pad(image, pad)
  gx = np.zeros_like(image)
  gy = np.zeros_like(image)
  
  for y in range(H):
    for x in range(W):
      gx[y, x] = np.sum(padded_image[y:y+ksize, x:x+ksize]*operator_x)
      gy[y, x] = np.sum(padded_image[y:y+ksize, x:x+ksize]*operator_y)
  return gx, gy

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

def build_Rtable(image, ksize=3):
  #以center point當作reference point
  H, W = image.shape
  center_point = (H//2, W//2)
  gx, gy = cal_gradient(image, SOBEL_GX, SOBEL_GY, ksize)
  gradient_directions = np.arctan2(gy, gx)

  r_table = {}
  for y in range(H):
    for x in range(W):
      if image[y, x] > 0: #if is edge point, cal point
        alpha = np.rad2deg(gradient_directions[y, x])
        #cal X' and Y'
        ry = y - center_point[0]
        rx = x - center_point[1]
        
        if alpha not in r_table:
          r_table[alpha] = []  
        r_table[alpha].append((ry, rx))

  return r_table

def detect_object(image, r_table, ksize=3, angle=90):
  gx, gy = cal_gradient(image, SOBEL_GX, SOBEL_GY, ksize)
  gradient_directions = np.arctan2(gy, gx)
  H, W = image.shape
  p_vote = np.zeros((*image.shape, len(range(0, 360, angle))), dtype=np.int32)
  
  #設定總共有幾種角度
  #EX: angle=90, 則會檢測0, 90, 180, 270的角度
  rad_angles = np.deg2rad(np.arange(0, 360, angle))
  cos_array = np.cos(rad_angles)
  sin_array = np.sin(rad_angles)
  
  for y in range(H):
    for x in range(W):
      if(image[y, x]) > 0: #if edge point, cal vote
        theta = np.rad2deg(gradient_directions[y, x])
        if theta in r_table:
          for ry, rx in r_table[theta]:
            # 計算選轉之後edge point到reference point之距離
            x_cs = x-(rx*cos_array+ry*sin_array)
            y_cs = y+(rx*sin_array-ry*cos_array)

            angle_idxs = np.arange(0, len(rad_angles))
            for i in range(len(x_cs)):
              val = (x_cs[i]>=0) and (x_cs[i]<W) and (y_cs[i]>=0) and (y_cs[i]<H)
              if val == True:
                p_vote[int(y_cs[i]), int(x_cs[i]), angle_idxs[i]]+=1            
  return p_vote

def show_result(reference, template, p_vote, ref_name, template_name, angle=90):
  box_H, box_W = template.shape
  # 獲取最多vote的reference point
  y, x, angle_idx = np.unravel_index(np.argmax(p_vote), p_vote.shape)
  print(y, x, angle_idx)
  pic_angle = angle_idx*angle
  result = reference.copy()
  
  fig, ax = plt.subplots()
  ax.imshow(result, cmap='gray')
  ax.plot( #畫出得到最多投票的reference point
    x, y,
    'ro',
    label=f"Point(x={x}, y={y}, \nangle={pic_angle})"
  )
  rect = patches.Rectangle(
    (x-box_W//2, y-box_H//2),
    box_W,
    box_H,
    linewidth=2,
    edgecolor='#32CD32',
    facecolor='none'
  )
  ax.add_patch(rect)
  ax.axis('off')
  ax.legend(loc='lower right', fontsize=12)
  inset_ax = fig.add_axes([0.01, 0.65 , 0.2, 0.2])  # 放置對比圖片
  inset_ax.imshow(template, cmap='gray')
  inset_ax.set_title('Template Image')
  inset_ax.axis("off")
  plt.savefig(f'{RESULT_ROOT_PATH}/{ref_name}_{template_name}_{pic_angle}.png')
  
def main():
  #template_list = [0, 45, 90, 120, 180, 315, 183, 327]
  #template_angle_list = [360, 45, 90, 120, 180, 15, 1, 1]
  
  template_list = [0, 33, 45, 177, 180, 240, 270, 315]
  template_angle_list = [360, 1, 45, 1, 180, 120, 90, 15]
  
  reference = cv2.imread('pic/Reference.png', cv2.IMREAD_GRAYSCALE)
  my_canny_reference = canny(reference)
  for template_angle, angle_step in zip(template_list, template_angle_list):
    start_time = time.time()
    template = cv2.imread(f'pic/Template_{template_angle}.png', cv2.IMREAD_GRAYSCALE)
    my_canny_template = canny(template)
    
    r_table = build_Rtable(my_canny_template, ksize=3)
    p_vote = detect_object(my_canny_reference, r_table=r_table, ksize=3, angle=angle_step)
    show_result(reference, template, p_vote, ref_name='manyItems', template_name='tea', angle=angle_step)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程式執行時間：{execution_time}s, Angle Step: {angle_step}")
    data = pd.DataFrame(p_vote[:, :, 0])
    
    fig = plt.figure()
    heatmap = sns.heatmap(data)
    fig.savefig(f'{RESULT_ROOT_PATH}/heatmap_tea_{template_angle}.png')
    #plt.savefig(f'{RESULT_ROOT_PATH}/heatmap_tea_{template_angle}.png')

if __name__ == '__main__':
  main()