import cv2 as cv

cam_feed = input("Enter video feed number or video filepath: ")

if cam_feed == '0':
    cap = cv.VideoCapture(int(cam_feed))
    while cap.isOpened():
        ret, img = cap.read()
        img = img.copy()

        cv.imshow('Human Pose Estimation with MoveNet', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

else:
    cap = cv.VideoCapture(cam_feed)
    while cap.isOpened():
        ret, img = cap.read()
        img = img.copy()
        img = cv.resize(img, (480, 690))

        cv.imshow('Human Pose Estimation with MoveNet', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
