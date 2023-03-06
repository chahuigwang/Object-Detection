import cv2
import numpy as np

def avg_frame(video): # 동영상의 모든 프레임을 averaging한 영상을 반환하는 함수
    cap = cv2.VideoCapture(video)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    sum_frame = np.zeros((height, width, 3), dtype=np.uint64)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                sum_frame += frame
            else:
                break
    cap.release()
    average = sum_frame / frames
    average = average.astype(np.uint8)
    return average, width, height, fps
