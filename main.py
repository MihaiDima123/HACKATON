from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import utils

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

utils.createTrackingWindow()

greenLower = (10,100,100)
greenUpper = (15,255,255)

pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    [l_h, l_s, l_v, u_h, u_s, u_v] = utils.getTrackbarPositions()

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    image = cv2.circle(frame, (0,0), 40, (l_h, l_s, l_v),32 )
    cv2.imshow('Tracking', image)

    image2 = cv2.circle(frame, (0,200), 40, (u_h, u_s, u_v),32 )
    cv2.imshow('Tracking', image2)

    mask = cv2.inRange(hsv, l_b, u_b)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    utils.drawCircles(frame, cnts)

    pts.appendleft(center)

    cv2.imshow("Frame", frame)
    cv2.imshow('Counturs', mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if not args.get("video", False):
	vs.stop()
else:
	vs.release()

cv2.destroyAllWindows()