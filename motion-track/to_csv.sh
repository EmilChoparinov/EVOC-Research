#!/bin/bash

# Path to the folder containing videos
video_folder="./videos"

# Loop through each video file in the folder
for video in "$video_folder"/*.{mp4,avi,mov,mkv}; do
    python3 videotocsv.py "$video"
done
