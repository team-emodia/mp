import cv2
import mediapipe as mp

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 인식할 관절 부위 정의
UPPER_BODY_LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
}

while cap.isOpened():
    # 웹캠에서 프레임 읽기
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # 성능 향상을 위해 이미지를 읽기 전용으로 표시
    image.flags.writeable = False
    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pose 모델로 이미지 처리
    results = pose.process(image)

    # 다시 BGR로 변환하여 화면에 표시
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 관절 감지 시
    if results.pose_landmarks:
        # 지정된 7개 부위만 화면에 표시
        for name, landmark_index in UPPER_BODY_LANDMARKS.items():
            landmark = results.pose_landmarks.landmark[landmark_index]
            
            # 랜드마크 좌표를 이미지 좌표로 변환
            h, w, c = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # 관절 위치에 원 그리고 이름 표시
            cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            cv2.putText(image, name, (cx - 20, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 터미널에 좌표 출력 (x, y, z, visibility)
            print(f'{name}: (x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}, visibility={landmark.visibility:.3f})')


    # 결과 이미지 보여주기
    cv2.imshow('MediaPipe Pose - Upper Body', image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()