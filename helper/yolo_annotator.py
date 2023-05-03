# annotates a single image
import cv2
import numpy as np
import json
from pathlib import Path
import random

# only need to edit here
folder = Path("/home/wenyi/DATA/XFS/SV/Figure11/yolo-format-resized/")
name = "bg5_0"
img_suffix = ".png"

label_path = folder / "labels" / (name + ".txt")
img_path = folder / "images" / (name + img_suffix)
print(label_path)
print(img_path)


colour = (255, 255, 255)

with open(label_path) as f:
  label_data = f.readlines()

image = cv2.imread(str(img_path))
img_height, img_width, _ = image.shape
for label in label_data:
  category, xc, yc, w, h = label.split()
  width = float(w) * img_width
  height = float(h) * img_height
  x = float(xc) * img_width - width / 2.0
  y = float(yc) * img_height - height / 2.0
  cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), colour, 1)
  # cv2.putText(image, category, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
# cv2.imwrite("/home/wenyi/DATA/synthetics/eg16.png", image)
cv2.imshow('image', image)
cv2.waitKey()
