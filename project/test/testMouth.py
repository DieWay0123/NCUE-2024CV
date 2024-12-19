import cv2
import mediapipe as mp

# 初始化 Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 嘴巴關鍵點索引
#OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]



# 打開攝像頭
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("忽略空幀")
        continue

    # 將影像轉為 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 將影像傳入 Mediapipe 處理
    results = face_mesh.process(image)

    # 還原為 BGR 顯示
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取嘴巴區域的關鍵點
            h, w, _ = image.shape
            for idx in INNER_LIP:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # 選擇性繪製完整臉部網格（可選）
            # mp_drawing.draw_landmarks(
            #     image, face_landmarks, mp_face_mesh.FACEMESH_LIPS,
            #     mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            #     mp_drawing_styles.get_default_face_mesh_contours_style()
            # )

    # 顯示結果
    cv2.imshow('Mediapipe Mouth Detection', image)

    # 按 Q 鍵退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()