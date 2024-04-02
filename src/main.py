import cv2 as cv
import detector
from src import config


def main():
    cam_feed = input("Enter video feed number or video filepath: ")
    if cam_feed == '0':
        cap = cv.VideoCapture(int(cam_feed))
        while cap.isOpened():
            ret, img = cap.read()

            img = img.copy()
            detect_pose = detector.PoseDetector()
            # Detect Pose
            keypoints = detect_pose.findkeypoints(img)
            # Rendering
            detect_pose.drawConnections(img, config.DetectorConfig.EDGES,
                                        keypoints,
                                        config.DetectorConfig.confidence_threshold)
            detect_pose.drawkeypoints(img, keypoints,
                                      config.DetectorConfig.confidence_threshold)

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
            detect_pose = detector.PoseDetector()
            # Detect Pose
            keypoints = detect_pose.findkeypoints(img)
            # Rendering
            detect_pose.drawConnections(img, config.DetectorConfig.EDGES,
                                        keypoints,
                                        config.DetectorConfig.confidence_threshold)
            detect_pose.drawkeypoints(img, keypoints,
                                      config.DetectorConfig.confidence_threshold)

            cv.imshow('Human Pose Estimation with MoveNet', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
