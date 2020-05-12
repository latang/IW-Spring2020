#-----------------------------------------------------------------------
# functions.py
# Author: Lauren Tang
#-----------------------------------------------------------------------

from scipy.spatial import distance as dist

# Credit to Adrian Rosebrock's blog for this function
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
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

# Adapted from Akshay L Chandra
# https://towardsdatascience.com/mouse-control-facial-movements-hci-app-c16b0494a971
# Return direction given the nose and anchor points.
def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point

    if nx < x - multiple * w and ny > y + multiple * h:
        return 'leftdown'
    elif nx < x - multiple * w and ny < y - multiple * h:
        return  'leftup'
    elif nx > x + multiple * w and ny > y + multiple * h:
        return 'rightdown'
    elif nx > x + multiple * w and ny < y - multiple * h:
        return 'rightup'

    if nx > x + multiple * w:
        return 'right'
    elif nx < x - multiple * w:
        return 'left'

    if ny > y + multiple * h:
        return 'down'
    elif ny < y - multiple * h:
        return 'up'

    return '-'