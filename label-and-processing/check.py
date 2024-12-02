import cv2
import os
import argparse
import json

from dataclasses import dataclass

@dataclass
class ColorSet():
    blue  = (0, 255, 0)
    red   = (0, 0, 255)
    white = (255, 255,255)

@dataclass
class SampleLabelColorSet():
    head  = ColorSet.white
    joint = ColorSet.blue
    box   = ColorSet.red

parser = argparse.ArgumentParser()

parser.add_argument("img_file", type=str, help="The file to load into the viewer")

# Collect names
img_file: str = parser.parse_args().img_file
json_file = img_file.replace('.rgb.jpg', '.json')

# Collect data
img = cv2.imread(img_file)
with open(json_file) as f: json_data = json.load(f)

colors_by_idx = [
    SampleLabelColorSet.head, 
    SampleLabelColorSet.joint, 
    SampleLabelColorSet.box
]


[[cv2.circle(img, (point[0], point[1]), 3, color, -1) for point in point_set["projected_location"]] 
    for point_set, color in 
            zip(json_data["objects"][0]["keypoints"], colors_by_idx)]


cv2.imshow("Checking Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()