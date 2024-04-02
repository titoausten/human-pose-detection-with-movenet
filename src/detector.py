import numpy as np
import cv2
import tensorflow as tf
import config


class PoseDetector:
    def __init__(self, model_path: str = config.DataConfig.model_path):
        self.keypoints_with_scores = None
        self.model_path = model_path

        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

    def findkeypoints(self, img):
        # Reshape Image to (192, 192)
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Make predictions
        self.interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        self.interpreter.invoke()

        # Results
        self.keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        return self.keypoints_with_scores

    def drawkeypoints(self, img, keypoints, confidence_threshold):
        height, width, channel = img.shape
        # Convert list of keypoints to 1D Array and multiplied by image pixels
        shaped_keypoints = np.squeeze(np.multiply(keypoints, [height, width, 1]))

        for keypoint in shaped_keypoints:
            keypoint_height, keypoint_width, keypoint_confidence = keypoint
            if keypoint_confidence > confidence_threshold:
                cv2.circle(img, (int(keypoint_width), int(keypoint_height)), 4,
                           (0, 255, 0), -1)

    def drawConnections(self, img, edges, keypoints, confidence_threshold):
        height, width, channel = img.shape
        # Convert list of keypoints to 1D Array and multiplied by image pixels
        shaped = np.squeeze(np.multiply(keypoints, [height, width, 1]))

        for edge, colour in edges.items():
            point1, point2 = edge
            height1, width1, confidence1 = shaped[point1]
            height2, width2, confidence2 = shaped[point2]

            if (confidence1 > confidence_threshold) & (confidence2 > confidence_threshold):
                cv2.line(img, (int(width1), int(height1)), (int(width2), int(height2)),
                         (0, 0, 255), 2)
