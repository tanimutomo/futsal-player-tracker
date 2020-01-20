from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from detect2 import YOLO, draw_bbox


config_path='config/yolov3.cfg'
weights_path='weights/yolov3.weights'
class_path='data/coco.names'
img_size=416 
conf_thres=0.8
nms_thres=0.4
# Load model and weights
model = Darknet(config_path, img_size=416).to("cpu")
model.load_darknet_weights(weights_path)
# model.cuda()
model.eval()
classes = utils.load_classes(class_path)
# Tensor = torch.cuda.FloatTensor
Tensor = torch.Tensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 0.8, 0.4)
    return detections[0]

class PointList():
    def __init__(self, npoints):
        self.npoints = npoints
        self.ptlist = np.empty((npoints, 2), dtype=int)
        self.pos = 0

    def add(self, x, y):
        if self.pos < self.npoints:
            self.ptlist[self.pos, :] = [x, y]
            self.pos += 1
            return True
        return False


def onMouse(event, x, y, flag, params):
    wname, img, ptlist = params
    if event == cv2.EVENT_MOUSEMOVE:  # マウスが移動したときにx線とy線を更新する
        img2 = np.copy(img)
        h, w = img2.shape[0], img2.shape[1]
        cv2.line(img2, (x, 0), (x, h - 1), (255, 0, 0))
        cv2.line(img2, (0, y), (w - 1, y), (255, 0, 0))
        cv2.imshow(wname, img2)

    if event == cv2.EVENT_LBUTTONDOWN:  # レフトボタンをクリックしたとき、ptlist配列にx,y座標を格納する
        if ptlist.add(x, y):
            print('[%d] ( %d, %d )' % (ptlist.pos - 1, x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), 3)
            cv2.imshow(wname, img)
        else:
            print('All points have selected.  Press ESC-key.')
        if(ptlist.pos == ptlist.npoints):
            print(ptlist.ptlist)
            cv2.line(img, (ptlist.ptlist[0][0], ptlist.ptlist[0][1]),
                     (ptlist.ptlist[1][0], ptlist.ptlist[1][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[1][0], ptlist.ptlist[1][1]),
                     (ptlist.ptlist[2][0], ptlist.ptlist[2][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[2][0], ptlist.ptlist[2][1]),
                     (ptlist.ptlist[3][0], ptlist.ptlist[3][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[3][0], ptlist.ptlist[3][1]),
                     (ptlist.ptlist[4][0], ptlist.ptlist[4][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[4][0], ptlist.ptlist[4][1]),
                     (ptlist.ptlist[0][0], ptlist.ptlist[0][1]), (0, 255, 0), 3)



# videopath = 'IMG_4904.MOV'
videopath = 'videoplayback.mp4'

import cv2
from IPython.display import clear_output
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
# initialize Sort object and video capture
from sort import *
vid = cv2.VideoCapture(videopath)
ret, frame = vid.read()
img = frame
wname = "MouseEvent"
cv2.namedWindow(wname)
npoints = 5
ptlist = PointList(npoints)
cv2.setMouseCallback(wname, onMouse, [wname, img, ptlist])
cv2.waitKey()
cv2.destroyAllWindows()
mot_tracker = Sort()

while(True):
    for ii in range(40):
        ret, frame = vid.read()
        cv2.line(img, (ptlist.ptlist[0][0], ptlist.ptlist[0][1]),
                 (ptlist.ptlist[1][0], ptlist.ptlist[1][1]), (0, 255, 0), 3)
        cv2.line(img, (ptlist.ptlist[1][0], ptlist.ptlist[1][1]),
                 (ptlist.ptlist[2][0], ptlist.ptlist[2][1]), (0, 255, 0), 3)
        cv2.line(img, (ptlist.ptlist[2][0], ptlist.ptlist[2][1]),
                 (ptlist.ptlist[3][0], ptlist.ptlist[3][1]), (0, 255, 0), 3)
        cv2.line(img, (ptlist.ptlist[3][0], ptlist.ptlist[3][1]),
                 (ptlist.ptlist[4][0], ptlist.ptlist[4][1]), (0, 255, 0), 3)
        cv2.line(img, (ptlist.ptlist[4][0], ptlist.ptlist[4][1]),
                 (ptlist.ptlist[0][0], ptlist.ptlist[0][1]), (0, 255, 0), 3)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)
        out_box_indices = []
        detections = np.array(detections)
        print('ptlist.ptlist:', ptlist.ptlist)
        for i, corrdinates in enumerate(detections):
            # corrdinates = corrdinates.tolist()
            pt = (int(corrdinates[2]), int(corrdinates[3]))
            print('pt: ', pt)
            result =  cv2.pointPolygonTest(ptlist.ptlist, pt, False)
            print('result:', result)
            if result >= 0:
                out_box_indices.append(i)
        print("-----")
        sys.exit(0)

        for i in out_box_indices:
            print(detections[i])
            detections = np.delete(detections, i, 0)
        print("-----")
        print(detections)
        detections = torch.tensor(detections)

        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0)*(img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0)*(img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            # for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            for x1, y1, x2, y2, obj_id in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                # cls = classes[int(cls_pred)]
                cls = classes[int(0)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h),
                             color, 4)
                # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60,
                #              y1), color, -1)
                # cv2.putText(frame, cls + "-" + str(int(obj_id)),
                #             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (255,255,255), 3)
        cv2.imshow("mot_tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        # fig=plt.figure(figsize=(12, 8))
        # plt.title("Video Stream")
        # plt.imshow(frame)
        # plt.show()
        # clear_output(wait=True)
