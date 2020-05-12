#!/usr/bin/env python

#-----------------------------------------------------------------------
# nosepointer.py
# Author: Lauren Tang
#-----------------------------------------------------------------------

import cv2
import dlib
from imutils import face_utils
import imutils
import pyautogui
import numpy as np
import time

default_detector = dlib.get_frontal_face_detector()
default_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

CHIN = 8
LEFT_EYE = 36
RIGHT_EYE = 45
LEFT_MOUTH = 48
RIGHT_MOUTH = 54
NOSE = 30

def nose_point(img, shape_predictor, detector):
    size = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    shape_exists = False

    # Detect faces
    for rect in rects:
        shape_exists = True
        shape = shape_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    if shape_exists == False:
        return img, None

    return (shape[NOSE][0], shape[NOSE][1])



# Adapted from Satya Mallick
# https://www.learnopencv.com/head-pose-estimation-
# using-opencv-and-dlib/?fbclid=IwAR1ZCNXi7MzWfZUTeXvqv
# FaOydeMs7cb2EVm9VgkHBmPfMGhMXJgzZGMigQ#code
def nose_pointer(img, shape_predictor, detector):
    # Read Image
    # im = cv2.imread("headPose.jpg");

    font = cv2.FONT_HERSHEY_SIMPLEX

    # if (detector == None):
    #     detector = default_detector
    #
    # if (shape_predictor == None):
    #     predictor = default_predictor
    # else:
    #     predictor =  dlib.shape_predictor(shape_predictor)

    size = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)

    shape_exists = False

    # Detect faces
    for rect in rects:
        shape_exists = True
        shape = shape_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    if shape_exists == False:
        return img, None

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        (shape[NOSE][0], shape[NOSE][1]),  # Nose tip
        (shape[CHIN][0], shape[CHIN][1]),  # Chin
        (shape[LEFT_EYE][0], shape[LEFT_EYE][1]),  # Left eye left corner
        (shape[RIGHT_EYE][0], shape[RIGHT_EYE][1]),  # Right eye right corne
        (shape[LEFT_MOUTH][0], shape[LEFT_MOUTH][1]),  # Left Mouth corner
        (shape[RIGHT_MOUTH][0], shape[RIGHT_MOUTH][1])  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    # print("Rotation Vector:\n {0}".format(rotation_vector))
    # print("Translation Vector:\n {0}".format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose


    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                                                     camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(img, p1, p2, (255, 0, 0), 2)

    return img, p2

if __name__ == '__main__':
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    video_capture = cv2.VideoCapture(0)
    origin = (0,0)


    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        # start = time.time()
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=650)

        if ret == 0:
            break
        # frame = cv2.flip(frame, 1)
        # frame = vs.read()
        # start = time.time()
        image, points = nose_pointer(frame, default_predictor, default_detector)
        # functiontime = time.time() - start

        # start2 = time.time()
        if points != None:
            pyautogui.moveTo(points[0], points[1], duration=0.01)

        # functiontime2 = time.time() - start2

        # start3 = time.time()
        # err, blink_occured = detect_blink(frame, default_predictor, default_detector)
        # # functiontime3 = time.time() - start3
        #
        # if err == 0:
        #     if blink_occured == (False, False, True): # blink
        #         pyautogui.click()
        #     elif blink_occured == (True, False, False): # left wink
        #         pyautogui.
        #     elif blink_occured == ( False, True, False): # right wink
        #         pyautogui.rightClick()

        cv2.rectangle(frame, (-60,-35), (60,35), (255,255,255), 2)


        # print("nosepoint {:.2f} movemouse {:.2f} blinkdetect{:.2f} click {:.2f}".
        #       format(functiontime, functiontime2, functiontime3, functiontime4))
        # Display image
        cv2.imshow("Output", image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
