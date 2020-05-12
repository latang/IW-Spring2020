# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from collections import deque

pastEar = deque([], 60)
pastEarDiff = deque([], 60)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# Constants
EYE_AR_THRESH = 0.19
EYE_AR_CONSEC_FRAMES = 15
WINK_AR_THRESH = 0.04
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10

# Counters
COUNTER = 0
TOTAL = 0
WINK_COUNTER = 0

TOTAL_LEFT = 0
TOTAL_RIGHT = 0

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # [42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # [36, 42)


def detect_blink(img, shape_predictor, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    shape_exists = False

    # Detect faces
    for rect in rects:
        shape_exists = True
        shape = shape_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    if shape_exists == False:
        return 1, (False, False, False)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    pastEar.append(ear)
    pastEarDiff.append(diff_ear)

    if diff_ear > WINK_AR_DIFF_THRESH:
        if leftEAR < rightEAR:
            if leftEAR < EYE_AR_THRESH:
                WINK_COUNTER += 1
                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    TOTAL_LEFT += 1
                    # print("left wink")
                    WINK_COUNTER = 0
                    ret =  0, (True, False, False)

        elif leftEAR > rightEAR:
            if rightEAR < EYE_AR_THRESH:
                WINK_COUNTER += 1
                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    TOTAL_RIGHT += 1
                    # print("right wink")
                    WINK_COUNTER = 0
                    ret =  0, (False, True, False)
        else:
            WINK_COUNTER = 0
    else:
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear <= EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                # print("blink occurred")
                ret = 0, (False, False, True)

            # reset the eye frame counter
            COUNTER = 0

    return ret


    # if len(pastEar) != 5:
    #     return
    # else:
    #     if pastEar[4] < EYE_AR_THRESH and pastEar[3] < EYE_AR_THRESH and pastEar[2] < EYE_AR_THRESH:
    #         print("blink")
    #         return True
