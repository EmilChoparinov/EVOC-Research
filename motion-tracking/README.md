# Video to CSV Keypoint Extraction

This script processes video footage of a gecko robot, detects keypoints using a pre-trained DREAM network, and saves the results to a CSV file.

## Key Functions

### generate_belief_map_visualizations
- Generates visual representations of belief maps for detected keypoints.
- Used for debugging and visualization purposes.

### network_inference
- Main function that handles the inference process.
- Loads the pre-trained network, processes video frames, and saves results.

## Input Parameters

Modify the `Arguments` class to set these parameters:

| Parameter | Description |
|-----------|-------------|
| `input_params_path` | Path to the pre-trained network weights file (.pth) |
| `image_path` | Path to the input video file |
| `input_config_path` | Path to the network configuration file (.yaml). If not specified, it's derived from `input_params_path` |
| `save_dir` | Directory to save the output CSV file |
| `image_preproc_override` | Optional image preprocessing override |
| `keypoints_path` | Not used in current implementation |
| `save_name` | Not used in current implementation |

## Output

1. **CSV File** (`gecko_data.csv`):
   - Columns: frame_id, head, center, forward, box_1, box_2, box_3, box_4, box_5, box_6, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6
   - Each row represents data for one video frame

2. **Visual Preview**:
   - Real-time display of processed frames with detected keypoints

## Key Features

- Video frame extraction and processing
- Keypoint detection for head, box corners, and joints
- Calculation of center point and forward direction
- CSV output for further analysis
- Real-time visualization of detection results

## Usage

1. Set up the `Arguments` class with appropriate paths and settings
2. Run the script to process the video and generate the CSV file

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PIL
- PyTorch
- ruamel.yaml
- Custom DREAM network implementation

## Notes

- Ensure the DREAM network model and configuration are compatible with the script
- The script processes the entire video by default. Modify if specific frame ranges are needed
- Adjust the `Arguments` class to match your file paths and preferences before running


Here is the [video2csv.py](video2csv.py) code,
Now I want u you adjust this code so it can read some folders of videos and process them all and save the results in a single CSV file.
the csv file names should be the same as the video file names.
all csv should be saved in ./save_data/[folder_name]/[video_name].csv