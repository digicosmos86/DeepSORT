import cv2

video_file = cv2.VideoCapture("./test.mp4")

while True:
    ret, frame = video_file.read()

    if not ret:
        raise RuntimeError("Could not read frame")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_file.release()
cv2.destroyAllWindows()