import numpy as np
height = 3
width = 3
ksize = 3
pad_size = 1

gx_cur = np.zeros((3,3))
gy_cur = np.zeros((3,3))

operator_x = np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
])
operator_y = np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
])

padded_image = np.array([
  [0, 0, 0, 0, 0],
  [0, 1, 2, 3, 0],
  [0, 4, 5, 6, 0],
  [0, 7, 8, 9, 0],
  [0, 0, 0, 0, 0]
])

for y in range(height):
  for x in range(width):
    for i in range(ksize):
      cur_y = y+i-1+pad_size
      for j in range(ksize):
        cur_x = x+j-1+pad_size
        gx_cur[i, j] = operator_x[i, j]*padded_image[cur_y, cur_x]
        gy_cur[i, j] = operator_y[i, j]*padded_image[cur_y, cur_x]
    print(np.sum(gx_cur))
    print(np.sum(gy_cur))