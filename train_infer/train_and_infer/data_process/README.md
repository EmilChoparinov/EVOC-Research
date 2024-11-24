# Gecko Robot Image Processing and Labeling

This project contains two Python scripts for processing video footage of a gecko robot and labeling key points on the extracted frames.

## Scripts

1. `split-frames.py`: Extracts frames from video files.
2. `label.py`: Allows manual labeling of key points on the extracted frames.

## split-frames.py

This script extracts frames from video files at regular intervals.

### Key Features:
- Processes multiple video files from a specified directory.
- Extracts frames at a configurable interval (default: every 32nd frame).
- Saves extracted frames as JPEG images with sequential numbering.

### Usage:
1. Set the `videos_path` variable to the directory containing your video files.
2. Adjust the `skip_frame` variable to change the frame extraction interval.
3. Run the script: `python split-frames.py`.

### Parameters:
- `output_folder`: Directory where extracted frames are saved.
- `skip_frame`: Number of frames to skip between extractions.
- `videos_path`: Path to the directory containing input video files.

## label.py

This script provides a GUI for manually labeling key points on the extracted frames.

### Key Features:
- Displays extracted frames for labeling.
- Allows marking of three types of points: white (head), red (box), and green (joints).
- Saves labeled data in JSON format.
- Supports continuing labeling from where you left off.

### Usage:
1. Ensure the extracted frames are in the `./rgb` directory.
2. Run the script: `python label.py`.

3. Use mouse buttons to mark points:
- Left click: Mark red points (box)
- Right click: Mark green points (joints)
- Middle click: Mark white points (head)
4. Press 'ESC' to exit the labeling process.

### Parameters:
- `input_path`: Directory containing input images (default: './rgb').
- `output_path`: Directory for saving labeled images and JSON files.
- `output_path_2`: Additional directory for saving copies of labeled data.

## Output Format

The `label.py` script generates a JSON file for each labeled image with the following structure:

```json
{
"objects": [{
 "class": "gecko",
 "keypoints": [
   {"name": "head", "location": [...], "projected_location": [...]},
   {"name": "joint", "location": [...], "projected_location": [...]},
   {"name": "box", "location": [...], "projected_location": [...]}
 ]
}]
}