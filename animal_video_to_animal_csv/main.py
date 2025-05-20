import csv
import os

import cv2

from animal_video_to_animal_csv.video_to_frames import get_points_from_frame

"""
  A
BCDEF
  G
HIJKL
  M
"""
labels = [chr(ord('A') + i) for i in range(13)]
headers = ['Frame'] + labels
for i in range(7, 64):
    frame = cv2.imread(f"./Files/Frames/frame_{i}.png")
    points = get_points_from_frame(frame)

    row = [i]
    for (x, y) in points:
        row.append(f"({x}, {y})")

    with open("output_points.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if os.stat("output_points.csv").st_size == 0:
            writer.writerow(headers)
        writer.writerow(row)
