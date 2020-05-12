# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from pattern_detector import Blink, Wink_left, Wink_right


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

# using shoelace method
def eye_area(eye):
	area = 0

	for i in range(0, 6):
		# print(i)
		if i == 5:
			area += (eye[i][0]*eye[1][1]) - (eye[1][0]*eye[i][1])
		else:
			area += (eye[i][0]*eye[i+1][1]) - (eye[i+1][0]*eye[i][1])

	return 0.5 * abs(area)

# def PolygonArea(corners):
# 	n = len(corners) # of corners
# 	area = 0.0
#
# 	for i in range(n):
# 		j = (i + 1) % n
# 		area += corners[i][0] * corners[j][1]
# 		area -= corners[j][0] * corners[i][1]
# 	area = abs(area) / 2.0
# 	return area


def metric(eye, eyebrow, height, width):
	area = eye_area(eye)
	num = 2 * area

	# print("my area is")
	# print(area)
	# print("vs")
	# print(PolygonArea(eye))

	distance = dist.euclidean(eye[1], eyebrow[2]) + \
		  dist.euclidean(eye[2], eyebrow[3]) + \
		  dist.euclidean(eye[3], eyebrow[4])
	den = height * width * distance

	# print("metric is {}".format(num / den))
	# print(float(num) / float(den))
	return np.tanh(num /den)



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.19
EYE_AR_CONSEC_FRAMES = 10
WINK_AR_THRESH = 0.05
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

WINK_COUNTER = 0

COUNTER_LEFT = 0
TOTAL_LEFT = 0

COUNTER_RIGHT = 0
TOTAL_RIGHT = 0

# b = Blink(False, 0)
# wl = Wink_left(False, 0)
# wr = Wink_right(False, 0)
#
# pattern = [b, wl, wr]
# print(pattern)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #[42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #[36, 42)
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

frame_counter=0

# frames=[]
# left_eye_scores=[]
# right_eye_scores=[]
# left_counters=[]
# right_counters=[]
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	# start = time.time()
	if fileStream and not vs.more():
		break

	# frame_counter += 1
	# if frame_counter == 30:
	# 	frame_counter =0
	# print("frame #: ", end='')
	# print(frame_counter)

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	# frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections, this is a tuple
	for rect in rects:
		# print("Detection : Left: {} Top: {} Right: {} Bottom: {}".format(
		# 	rect.left(), rect.top(), rect.right(), rect.bottom()))

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		# print("right eye", rightEye)
		# print(type(rightEye), rightEye.shape)
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		diff_ear = np.abs(leftEAR - rightEAR)

	# FOR OTHER METHOD
		# leftEyebrow = shape[17:22]
		# rightEyebrow = shape[22:27]
		#
		# topleft = (rect.left(), rect.top())
		# bottomright = (rect.right(), rect.bottom())
		#
		# height = rect.right() - rect.left()
		# width = rect.bottom() - rect.top()
		# leftValue = metric(leftEye, leftEyebrow, height, width)
		# rightValue = metric(rightEye, rightEyebrow, height, width)

		# left_eye_scores.append(leftEAR)
		# right_eye_scores.append(rightEAR)

		# val = (leftValue+rightValue) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		# leftConvex = ConvexHull(leftEye)
		# rightConvex = ConvexHull(rightEye)
		# print("left {}".format(leftConvex.area))
		# print("right {}".format(rightConvex.area))

		# leftEyeHull = cv2.convexHull(leftEye)
		# rightEyeHull = cv2.convexHull(rightEye)
		# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# if one eye is blinking and other is not, its a wink.
		# if both eyes is below threshold, its a blink
		if diff_ear > WINK_AR_DIFF_THRESH:
			if leftEAR < rightEAR:
				if leftEAR < EYE_AR_THRESH:
					WINK_COUNTER += 1
					if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
						TOTAL_LEFT += 1
						print("left wink")
						WINK_COUNTER = 0

			elif leftEAR > rightEAR:
				if rightEAR < EYE_AR_THRESH:
					WINK_COUNTER += 1
					if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
						TOTAL_RIGHT += 1
						print("right wink")
						WINK_COUNTER = 0
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
					print("blink occurred")
				# reset the eye frame counter
				COUNTER = 0
			# else:
			# 	COUNTER = 0
			# 	WINK_COUNTER = 0


		# end = time.time()


		# if leftEAR < EYE_AR_THRESH:
		# 	COUNTER_LEFT += 1
		# else:
		# 	if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
		# 		TOTAL_LEFT += 1
		# 		print("hello")
		# 		wl.incrementCount()
		# 	# reset the eye frame counter
		# 	COUNTER_LEFT = 0
		#
		# if rightEAR < EYE_AR_THRESH:
		# 	COUNTER_RIGHT += 1
		# else:
		# 	if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
		# 		TOTAL_RIGHT += 1
		# 	# reset the eye frame counter
		# 	COUNTER_RIGHT = 0

		# if rightEAR < EYE_AR_THRESH:
		# 	COUNTER_RIGHT += 1
		# 	if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
		# 		TOTAL_RIGHT += 1
		# 	COUNTER_RIGHT = 0
		# elif leftEAR < EYE_AR_THRESH:
		# 	COUNTER_LEFT += 1
		# 	if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
		# 		TOTAL_LEFT += 1
		# 	COUNTER_LEFT = 0

		#right eye wink

		# if leftEAR > rightEAR + 0.05:
		# 	print("left wink in prog")
		# 	if leftEAR < EYE_AR_THRESH:
		# 		COUNTER_LEFT += 1
		# 		print("count left: ", end = '')
		# 		print(COUNTER_LEFT)
		# 	else:
		# 		if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
		# 			TOTAL_LEFT += 1
		# 			# print("hello")
		# 			print("left wink")
		# 			wl.incrementCount()
		# 		COUNTER_LEFT = 0
		# elif rightEAR < leftEAR + 0.05:
		# 	print("right wink in prog")
		# 	if rightEAR < EYE_AR_THRESH:
		# 		COUNTER_RIGHT += 1
		# 		print("count right", end = '')
		# 		print(COUNTER_RIGHT)
		# 	else:
		# 		if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
		# 			TOTAL_RIGHT += 1
		# 			print("right wink")
		# 		COUNTER_RIGHT = 0
		# left eye wink

		# print('get wl count')
		# wl.getCount()
		# if wl.getCount() == 3:
			# print('hello!')
			# cv2.putText(frame, "pls werk :(", (10, 60),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# wl.restartCount()

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		# cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		#
		# cv2.putText(frame, "Wink Left : {}".format(TOTAL_LEFT), (10, 60),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# cv2.putText(frame, "Wink Right: {}".format(TOTAL_RIGHT), (10, 90),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# fps = round(1.0/ (end-start), 1)

		color = (255, 0, 0)

		# frame = cv2.rectangle(frame, topleft, bottomright, color, 2)

		# cv2.putText(frame, "FPS: {:.1f}".format(fps), (50, 30),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "blink: {:.1f}".format(TOTAL), (50, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "left: {:.1f}".format(TOTAL_LEFT), (50, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "right: {:.1f}".format(TOTAL_RIGHT), (50, 90),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "Left: {:.4f}".format(leftEAR), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "Right: {:.4f}".format(rightEAR), (300, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	if key == ord("r"):
		TOTAL_RIGHT = 0
		TOTAL = 0
		TOTAL_LEFT = 0


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()