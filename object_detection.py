from avg_frame import *

file_path = '/Users/chahuigwang/PythonWorkspace/Object Detection/highway.mov' # 동영상 파일 경로
cap = cv2.VideoCapture(file_path)
bg_img, w, h, fps = avg_frame(file_path)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('object_detection.avi', fourcc, fps, (w, h))

bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY) # 배경 이미지를 gray scale 이미지로 변환
bg_img = cv2.GaussianBlur(bg_img, (0, 0), 1.0)

kernel10 = np.ones((10, 10), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

        diff = cv2.absdiff(gray, bg_img) # substraction의 절댓값
        _, th = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY) # 차이가 20 이상이면 255(흰색), 20보다 작으면 0(검정색)

        diff = cv2.morphologyEx(th, cv2.MORPH_BLACKHAT, kernel10)
        diff = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel5)

        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(th)

        for i in range(1, cnt):
            x, y, w, h, area = stats[i]

            if area < 50:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
    
        out.write(frame)
    else:
        break

cap.release()
out.release()
