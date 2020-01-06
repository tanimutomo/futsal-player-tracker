### import ###
from __future__ import division


# import python scripts
from models import *
from utils.utils import *
from utils.datasets import *


# python libraries
import os
import sys
import argparse # parser
from PIL import Image # image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import NullLocator



to_tensor = transforms.ToTensor() # PIL Image to torch.Tensor
to_pil = transforms.ToPILImage() # torch.Tensor to PIL Image

def object_detection(frame):
    # frame = frame.transpose()
    # frame = torch.from_numpy(frame) # [h, w, c] # uint8
    # print(frame.shape, frame.dtype, frame.min(), frame.max())

    frame = Image.fromarray(frame) # PIL Image
    frame = frame.resize((416, 416))
    frame = to_tensor(frame) # torch.Tensor # float32

    # Pad to square resolution
    # frame, _ = pad_to_square(frame, 0)
    # Resize
    # frame = resize(frame, 416)
    # print(frame.shape)
    frame = frame.unsqueeze(0)
    # print(frame.shape)
    # gpu or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model <--- YOLO
    model = Darknet("config/yolov3.cfg", img_size=416).to(device)

    # load trained parameters to model
    if "weights/yolov3.weights".endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights("weights/yolov3.weights")
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load("weights/yolov3.weights"))

    model.eval()  # Set in evaluation mode

    # Convert np.ndarray -> torch.Tensor
    #frame = ...(frame)

    # Get detections
    with torch.no_grad():
        detections = model(frame) # detect
        detections = non_max_suppression(detections, 0.8, 0.4) # post-process
    return detections



def draw_bbox(img, detections):
    classes = load_classes("data/coco.names")  # Extracts class labels from file

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    bbox_coordinate = []
    # print(detections)

    for img_i, (path, detections) in enumerate(zip(img, detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        # img = np.array(Image.open(path))
        # print(img.shape[:2])
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, 416, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                # print(x1, y1, box_w, box_h)
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

                bbox_coordinate = bbox_coordinate + [x1.tolist(), y1.tolist(), box_w.tolist(), box_h.tolist()]

                # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = "test1"
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
        # print(bbox_coordinate)
        return bbox_coordinate


if __name__ == '__main__':
 # python detect.py --image_folder hoge
     parser = argparse.ArgumentParser()
     parser.add_argument("--image_folder", type=str)
     parser.add_argument("--img_size", type=int, default=416)
     opt = parser.parse_args()
     # Extract image as PyTorch tensor
     frame = transforms.ToTensor()(Image.open(opt.image_folder))
     detections = object_detection(frame)
     print(detections)
