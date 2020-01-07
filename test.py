from imutils.video import VideoStream
from imutils.video import FPS
from utils.utils import *
from utils.datasets import *

from detect2 import YOLO, draw_bbox
import numpy as np
import argparse
import imutils
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(ap.parse_args())

(major, minor) = cv2.__version__.split(".")[:2]

if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
        }
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

initBB = None

if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

else:
    vs = cv2.VideoCapture(args["video"])

fps = None

yolo = YOLO()

count = 0
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    (H, W) = frame.shape[:2]

    if initBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (w, h),
                (0, 255, 0), 2)

        print("update")
        fps.update()
        fps.stop()

        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if count % 10 == 0:
        print("detection実行")

        detections = yolo.detect(frame)
        print(detections)
        print("↑detectionsだぞ")

        img = np.array(frame)
        initBB = draw_bbox(img, detections)
        print(type(initBB))
        initBB = list(initBB)
        tracker.init(frame, tuple(initBB[0:4]))

        while len(initBB) > 0:
            coordinate = initBB[0:4]
            del initBB[0:4]
            coordinate = tuple(coordinate)
            print(coordinate)


        fps = FPS().start()

    elif key == ord("q"):
        break
    count = count + 1


if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
