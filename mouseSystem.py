# USAGE
# python3.6 mouseSystem.py

#-----------------------------------------------------------------------
# mouseSystem.py
# Author: Lauren Tang

# Some portions inspired by Akshay L Chandra
# https://towardsdatascience.com/mouse-control-facial-movements-hci-app-c16b0494a971
# Some portions inspired by Adrian Rosebrock,
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
#-----------------------------------------------------------------------

# import the necessary packages
from functions import eye_aspect_ratio, direction
from imutils import face_utils
import imutils
import dlib
import cv2
import pyautogui
import numpy as np
from nosepointer import nose_point
import time

# Constants
EYE_AR_THRESH = 0.20
EAR_CONSEC_FRAMES = 10
WINK_AR_DIFF_THRESH = 0.04
WINK_CONSEC_FRAMES = 5

# initialize the frame counters and the total number of blinks
BLINK_COUNTER = 0
TOTAL = 0

WINK_COUNTER = 0

COUNTER_LEFT = 0
TOTAL_LEFT = 0

COUNTER_RIGHT = 0
TOTAL_RIGHT = 0

# CAM_WIDTH = 1280
# CAM_HEIGHT = 720

SCROLL_MODE = False

WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # [42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # [36, 42)
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))

if __name__ == '__main__':
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    video_capture = cv2.VideoCapture(0)
    start = True

    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        # start = time.time()
        ret, frame = video_capture.read()
        if ret == 0:
            break
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=550)

        # Detect faces in the grayscale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        # Loop over the face detections, this is a tuple
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Image flipped
            temp = leftEye
            leftEye = rightEye
            rightEye = temp

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the eye aspect ratio together for both eyes
            averageEAR = (leftEAR + rightEAR) / 2.0

            # Get difference in EAR value between eyes
            diffEAR = np.abs(leftEAR - rightEAR)

            # if one eye is blinking and other is not, its a wink.
            # if both eyes is below threshold, its a blink
            if diffEAR > WINK_AR_DIFF_THRESH:
                if leftEAR < rightEAR:
                    if leftEAR < EYE_AR_THRESH:
                        COUNTER_LEFT += 1
                    else:
                        if COUNTER_LEFT > WINK_CONSEC_FRAMES:
                            # TOTAL_LEFT += 1
                            print("left hold")
                            SCROLL_MODE = not SCROLL_MODE
                            COUNTER_LEFT = 0

                elif leftEAR > rightEAR:
                    if rightEAR < EYE_AR_THRESH:
                        COUNTER_RIGHT += 1
                    else:
                        if COUNTER_RIGHT > WINK_CONSEC_FRAMES:
                            TOTAL_RIGHT += 1
                            print("right wink")
                            pyautogui.click(button='right')
                            COUNTER_RIGHT = 0
                else:
                    COUNTER_RIGHT = 0
                    COUNTER_LEFT = 0
            else:
                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if averageEAR <= EYE_AR_THRESH:
                    BLINK_COUNTER += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if BLINK_COUNTER >= EAR_CONSEC_FRAMES:
                        TOTAL += 1
                        print("blink occurred")
                        pyautogui.click(button='left')
                        cv2.putText(frame, "Left Click", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

                    # reset the eye frame counter
                    BLINK_COUNTER = 0

        if SCROLL_MODE:
            cv2.putText(frame, "Scrolling on", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        else:
            cv2.putText(frame, "Scrolling off", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

        npoints = nose_point(frame, predictor, detector)

        w, h = 25, 20
        multiple = 1

        if npoints != None:
            if start:
                x, y = npoints
                start = False
            else:
                cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
                cv2.line(frame, (x,y), npoints, BLUE_COLOR, 2)

        dir = direction(npoints, (x, y), w, h)
        drag = 17
        scrollDrag = 1

        # Diagonal moves
        if dir == 'leftdown':
            if not SCROLL_MODE:
                pyautogui.moveRel(-drag, drag)
        elif dir == 'leftup':
            if not SCROLL_MODE:
                pyautogui.moveRel(-drag, -drag)
        elif dir == 'rightdown':
            if not SCROLL_MODE:
                pyautogui.moveRel(drag, drag)
        elif dir == 'rightup':
            if not SCROLL_MODE:
                pyautogui.moveRel(drag, -drag)
        # 4 directions
        elif dir == 'right':
            if SCROLL_MODE:
                pyautogui.hscroll(scrollDrag)
            else:
                pyautogui.moveRel(drag, 0)
        elif dir == 'left':
            if SCROLL_MODE:
                pyautogui.hscroll(-scrollDrag)
            else:
                pyautogui.moveRel(-drag, 0)
        elif dir == 'up':
            if SCROLL_MODE:
                pyautogui.scroll(scrollDrag)
            else:
                pyautogui.moveRel(0, -drag)
        elif dir == 'down':
            if SCROLL_MODE:
                pyautogui.scroll(-scrollDrag)
            else:
                pyautogui.moveRel(0, drag)

        # Display image
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
