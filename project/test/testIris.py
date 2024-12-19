import cv2

import mediapipe as mp

import numpy as np

import itertools

from joblib import load

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles #mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh

#訓練依虹膜寬度預測距離的模型

xx = np.array( [ 41.61639066,35.65641632,28.98277593,23.0578019,19.36509066,16.13595585,13.10885884,11.94330933,\

10.64142594,9.253465078,8.909845501,8.038228499,7.494420616 ]).reshape((-1, 1))

yy = np.array( [20,25,30,40,50,60,70,80,90,100,110,120,130] ).reshape((-1, 1))

sc_X = StandardScaler()

sc_y = StandardScaler()

x = sc_X.fit_transform(xx)

y = sc_y.fit_transform(yy)

dist_model = SVR(kernel=’rbf’)

dist_model.fit(x,y)

def predict_dist(d):

y_pred = sc_y.inverse_transform((dist_model.predict(sc_X.transform(np.array([[d]])))))

return y_pred

fliph_img = True

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(

max_num_faces=1,

refine_landmarks=True,

min_detection_confidence=0.5,

min_tracking_confidence=0.5) as face_mesh:

while cap.isOpened():

success, image = cap.read()

if not success:

print(“Ignoring empty camera frame.")

# If loading a video, use ‘break’ instead of ‘continue’.

continue

# Flip the image horizontally for a selfie-view display.

if fliph_img is True:

image = cv2.flip(image, 1)

# To improve performance, optionally mark the image as not writeable to

# pass by reference.

image.flags.writeable = False

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

(frame_h, frame_w, _) = image.shape

results = face_mesh.process(image)

# Draw the face mesh annotations on the image.

image.flags.writeable = True

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

if results.multi_face_landmarks:

iris_xy = []

for fid, face_landmarks in enumerate(results.multi_face_landmarks):


IRISES_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_IRISES)))

for eid, IRISE_INDEX in enumerate(IRISES_INDEXES):

ratio_x = float(face_landmarks.landmark[IRISE_INDEX].x)

ratio_y = float(face_landmarks.landmark[IRISE_INDEX].y)

ratio_z = float(face_landmarks.landmark[IRISE_INDEX].z)

px, py, pz = ratio_x*frame_w, ratio_y*frame_h, ratio_x

iris_xy.append((px, py))

mp_drawing.draw_landmarks(

image=image,

landmark_list=face_landmarks,

connections=mp_face_mesh.FACEMESH_IRISES,

landmark_drawing_spec=None,

connection_drawing_spec=mp_drawing_styles

.get_default_face_mesh_iris_connections_style())


if len(iris_xy)>=3:

left_iris_diameter = iris_xy[0][0] – iris_xy[2][0]

dist_left = round(predict_dist(left_iris_diameter)[0], 1)

cv2.putText(image, str(dist_left)+’cm’, ( 10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

if len(iris_xy)>=7:

right_iris_diameter = iris_xy[4][0] – iris_xy[6][0]

dist_right = round(predict_dist(right_iris_diameter)[0], 1)

cv2.putText(image, str(dist_right)+’cm’, ( 130,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

cv2.imshow(‘MediaPipe Face Mesh’, image)

k = cv2.waitKey(1)

if k == 113:

break

out.release()

cap.release()