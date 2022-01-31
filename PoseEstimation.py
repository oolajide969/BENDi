from pprint import pprint

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle_here = np.abs(radians * 180.0 / np.pi)

    if angle_here > 180.0:
        angle_here = 360 - angle_here

    return angle_here


bodyAngles = {}
# For webcam input:
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("1.mp4")
# For Video input:
prevTime = 0
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")

            # If loading a video, use 'break' instead of 'continue'.
            break

        # Recoloring image from BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # detection of points is done here
        results = pose.process(image)

        # Recoloring image from RGB image to BGR.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extracting landmarks
        try:

            landmarks = results.pose_landmarks.landmark

            # Storing points into variables
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

            # calculating the angles on each point
            leftWristAngle = calculate_angle(left_elbow, left_wrist, left_index)
            leftElbowAngle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            leftShoulderAngle = calculate_angle(left_hip, left_shoulder, left_elbow)
            leftHipAngle = calculate_angle(left_knee, left_hip, left_shoulder)
            leftKneeAngle = calculate_angle(left_ankle, left_knee, left_hip)
            leftAnkleAngle = calculate_angle(left_foot_index, left_knee, left_hip)
            rightWristAngle = calculate_angle(right_elbow, right_wrist, right_index)
            rightElbowAngle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            rightShoulderAngle = calculate_angle(right_hip, right_shoulder, right_elbow)
            rightHipAngle = calculate_angle(right_knee, right_hip, right_shoulder)
            rightKneeAngle = calculate_angle(right_ankle, right_knee, right_hip)
            rightAnkleAngle = calculate_angle(right_foot_index, right_knee, right_hip)

            angles = [leftWristAngle, leftElbowAngle, leftShoulderAngle, leftHipAngle, leftKneeAngle, leftAnkleAngle,
                      rightWristAngle, rightElbowAngle, rightShoulderAngle, rightHipAngle, rightKneeAngle,
                      rightAnkleAngle]

            angleStrings = ["leftWristAngle", "leftElbowAngle", "leftShoulderAngle", "leftHipAngle", "leftKneeAngle",
                            "leftAnkleAngle", "rightWristAngle", "rightElbowAngle", "rightShoulderAngle",
                            "rightHipAngle", "rightKneeAngle", "rightAnkleAngle"]

            # for angle in angles:
            #     newAngle = string(angle)
            #     angleStrings.append(newAngle)

            bodyAngles = dict(zip(angleStrings, angles))
            pprint(bodyAngles, width=1)
        except:
            print("i don come")
            pass

        # landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] gives coordinate
        # mp_pose.PoseLandmark.LEFT_SHOULDER.value gives index of landmark in the PoseLandmark array
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # currTime = time.time()
        # fps = 1 / (currTime - prevTime)
        # prevTime = currTime
        # cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

        cv2.imshow('MediaPipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
