import asyncio
import base64
import logging
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from PIL import Image

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 인식할 관절 부위 정의
UPPER_BODY_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

app = FastAPI()
sio = socketio.AsyncServer(async_mode="asgi")
app.mount("/socket.io", socketio.ASGIApp(sio))

logging.basicConfig(level=logging.INFO)


def process_image(image_data):
    """웹캠 이미지 처리"""
    try:
        # Base64 -> Image
        image_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)


        # 성능 향상을 위해 이미지를 읽기 전용으로 표시
        frame.flags.writeable = False
        # BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose 모델로 이미지 처리
        results = pose.process(image_rgb)

        # 다시 BGR로 변환하여 화면에 표시
        frame.flags.writeable = True

        # 관절 감지 시
        if results.pose_landmarks:
            # 지정된 7개 부위만 화면에 표시
            for name, landmark_index in UPPER_BODY_LANDMARKS.items():
                landmark = results.pose_landmarks.landmark[landmark_index]

                # 랜드마크 좌표를 이미지 좌표로 변환
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # 관절 위치에 원 그리고 이름 표시
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                cv2.putText(
                    frame,
                    name,
                    (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # 터미널에 좌표 출력 (x, y, z, visibility)
                print(
                    f"{name}: (x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}, visibility={landmark.visibility:.3f})"
                )
        # OpenCV 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@sio.on("connect")
async def connect(sid, environ):
    logging.info(f"New client connected: {sid}")


@sio.on("disconnect")
async def disconnect(sid):
    logging.info(f"Client disconnected: {sid}")


@sio.on("frame")
async def handle_frame(sid, data):
    """Handle webcam frame from client."""
    processed_image = process_image(data)
    if processed_image:
        await sio.emit("processed_frame", processed_image, to=sid)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
