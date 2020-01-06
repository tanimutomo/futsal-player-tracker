#import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from utils.utils import *
from utils.datasets import *

from detect2 import object_detection, draw_bbox
import numpy as np
import argparse
import imutils
import time
import cv2


#construct the argument pasere and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
        }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # tracker1 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # tracker2 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # tracker3 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # tracker4 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # tracker5 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # tracker6 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # tracker7 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None

count = 0
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # (success, box1) = tracker1.update(frame)
        # (success, box2) = tracker2.update(frame)
        # (success, box3) = tracker3.update(frame)
        # (success, box4) = tracker4.update(frame)
        # (success, box5) = tracker5.update(frame)
        # (success, box6) = tracker6.update(frame)
        # (success, box7) = tracker7.update(frame)


        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (w, h),
                (0, 255, 0), 2)

            # (x, y, w, h) = [int(v) for v in box1]
            # print(x, y, w, h)
            # cv2.rectangle(frame, (x, y), (w, h),
            #     (255, 255, 0), 2)
            #
            # (x, y, w, h) = [int(v) for v in box2]
            # print(x, y, w, h)
            # cv2.rectangle(frame, (x, y), (w, h),
            #     (255, 255, 255), 2)
            #
            # (x, y, w, h) = [int(v) for v in box3]
            # print(x, y, w, h)
            # cv2.rectangle(frame, (x, y), (w, h),
            #     (255, 0, 0), 2)
            #
            # (x, y, w, h) = [int(v) for v in box4]
            # print(x, y, w, h)
            # cv2.rectangle(frame, (x, y), (w, h),
            #     (0, 0, 255), 2)
            #
            # (x, y, w, h) = [int(v) for v in box5]
            # print(x, y, w, h)
            # cv2.rectangle(frame, (x, y), (w, h),
            #     (255, 0, 255), 2)
            #
            # (x, y, w, h) = [int(v) for v in box6]
            # print(x, y, w, h)
            # cv2.rectangle(frame, (x, y), (w, h),
            #     (0, 0, 0), 2)
            #
            # (x, y, w, h) = [int(v) for v in box7]
            # print(x, y, w, h)
            # cv2.rectangle(frame, (x, y), (w, h),
            #     (0, 0, 0), 2)
            #
            # (x, y, w, h) = [int(v) for v in box]
            # cv2.rectangle(frame, (x, y), (x + w, y + h),
            #     (0, 255, 0), 2)
            #
        # update the FPS counter
        print("update")
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if count % 10 == 0:
        print("detection実行")

    # if key == ord("s"):
        detections = object_detection(frame)
        print(detections)
        print("↑detectionsだぞ")

        img = np.array(frame)
        initBB = draw_bbox(img, detections)
        print(type(initBB))
        initBB = list(initBB)
        # print(initBB)
        tracker.init(frame, tuple(initBB[0:4]))
        # tracker1.init(frame, tuple(initBB[4:8]))
        # tracker2.init(frame, tuple(initBB[8:12]))
        # tracker3.init(frame, tuple(initBB[12:16]))
        # tracker4.init(frame, tuple(initBB[16:20]))
        # tracker5.init(frame, tuple(initBB[20:24]))
        # tracker6.init(frame, tuple(initBB[24:28]))
        # tracker7.init(frame, tuple(initBB[24:28]))

        while len(initBB) > 0:
            coordinate = initBB[0:4]
            del initBB[0:4]
            coordinate = tuple(coordinate)
            print(coordinate)


        # initBB = object_detection(frame)
        # print(initBB)
        # d_array = []
        # for f in initBB:
        #     d_array.append(f[0])
        #     print(d_array)
        # for coordinate in d_array:
        #     print(coordinate[0])
        #     coordinate = coordinate
        # initBB = coordinate[0:4]


            # print(initBB[0][i][:4])
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        # initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
        # print(initBB)
        # tracker.init(frame, initBB)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
    count = count + 1


# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
