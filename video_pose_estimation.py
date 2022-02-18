from pprint import pprint

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def calculate_angle(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    radians = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] -
                                                                                    point2[0])
    angle_here = np.abs(radians * 180.0 / np.pi)

    if angle_here > 180.0:
        angle_here = 360 - angle_here

    return angle_here


bodyAngles = {}
instructorBodyAngle = {'leftAnkleAngle': 172.38870036139213,
                       'leftElbowAngle': 171.73912041407684,
                       'leftHipAngle': 178.04032738004167,
                       'leftKneeAngle': 179.12044364492928,
                       'leftShoulderAngle': 175.78916004181852,
                       'leftWristAngle': 169.61686756776467,
                       'rightAnkleAngle': 101.35695755419812,
                       'rightElbowAngle': 159.1497048400839,
                       'rightHipAngle': 132.55681087015446,
                       'rightKneeAngle': 85.21983632190597,
                       'rightShoulderAngle': 165.75220502202595,
                       'rightWristAngle': 177.44200074079131}
good_counter = 0
off_counter = 0

# For webcam input:
cap = cv2.VideoCapture(0)
# For Video input:
# cap = cv2.VideoCapture("1.mp4")


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

            # check if all points are in frame
            # storing visibility points
            leftShoulderVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
            leftElbowVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
            leftWristVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility
            leftHipVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
            leftKneeVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            leftAnkleVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
            leftIndexVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].visibility
            leftFootVisibilityPoint = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
            rightShoulderVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
            rightElbowVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            rightWristVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
            rightHipVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            rightKneeVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            rightAnkleVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility
            rightIndexVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].visibility
            rightFootVisibilityPoint = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility

            visibilityPoints = [rightIndexVisibilityPoint, rightHipVisibilityPoint, rightAnkleVisibilityPoint,
                                rightKneeVisibilityPoint, rightElbowVisibilityPoint, rightWristVisibilityPoint,
                                rightShoulderVisibilityPoint, rightFootVisibilityPoint, leftIndexVisibilityPoint,
                                leftHipVisibilityPoint, leftAnkleVisibilityPoint, leftKneeVisibilityPoint,
                                leftElbowVisibilityPoint, leftWristVisibilityPoint, leftShoulderVisibilityPoint,
                                leftFootVisibilityPoint]
            pointStrings = ["right_index", "right_hip", "right_ankle", "right_knee", "right_elbow", "right_wrist",
                            "right_shoulder", "right_foot_index", "left_index", "left_hip", "left_ankle", "left_knee",
                            "left_elbow", "left_wrist", "left_shoulder", "left_foot"]
            bodyPoints = dict(zip(pointStrings, visibilityPoints))

            for pointStrings in bodyPoints:
                if (bodyPoints[pointStrings]) >= 0.6:
                    print(f"{pointStrings}:{bodyPoints[pointStrings]} is in frame")
                elif (bodyPoints[pointStrings]) >= 0.35 and (bodyPoints[pointStrings]) < 0.6:
                    print(f"{pointStrings}:{bodyPoints[pointStrings]} might be skewed or covered")
                else:
                    print(f"{pointStrings}:{bodyPoints[pointStrings]} is not well placed in frame")

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

            bodyAngles = dict(zip(angleStrings, angles))

        except:
            print("Halt!")
            pass

        # landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] gives coordinate
        # mp_pose.PoseLandmark.LEFT_SHOULDER.value gives index of landmark in the PoseLandmark array
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
print("")
pprint(bodyAngles, width=1)
print("")

for angle in instructorBodyAngle:
    for angles in bodyAngles:
        if (instructorBodyAngle[angle] == bodyAngles[angle]) or (
                round(abs(instructorBodyAngle[angle] - bodyAngles[angle]), 2) <= 10):
            print(angle, "is aligned")
            good_counter += 1
            break
        else:
            print(angle, "is off by", round(abs(instructorBodyAngle[angle] -
                                                bodyAngles[angle]), 2), "degrees")
            off_counter += 1
            break
print(" ")
print(good_counter)
print(off_counter)

if good_counter >= 10:
    print("Job well done!")
elif (good_counter >= 5) and (good_counter <= 9):
    print("Good job, we go again!")
else:
    print("common! this is not it")
