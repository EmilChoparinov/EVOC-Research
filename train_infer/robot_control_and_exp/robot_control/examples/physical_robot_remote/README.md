# Comprehensive Guide: Physical Modular Robot Remote Control and Optimization

main.py

## Overview

This script demonstrates how to remotely control a physical modular robot using a CPG (Central Pattern Generator) network brain, record its movements, analyze its performance, and optimize its behavior. The system includes:

1. Robot body structure definition
2. CPG network brain creation and configuration
3. Remote control setup and execution
4. Video recording of robot movement
5. Computer vision-based movement analysis
6. Iterative optimization of CPG network weights

## Key Components and Detailed Workflow

### 1. Imports and Dependencies

The script relies on several libraries:
- `revolve2`: For modular robot definitions and control
- `numpy`: For numerical operations
- `cv2` (OpenCV): For video capture and processing
- `torch`: For neural network operations (DREAM network)
- `multiprocessing`: For concurrent video recording
- `ruamel.yaml`: For YAML file parsing
- `PIL`: For image processing
- `json`: For data serialization
- `sklearn.cluster`: For K-means clustering (unused in the current version)

### 2. Robot Body Definition

Two functions are provided to create robot bodies:

#### `make_body()` function
- Creates a simple robot body with 4 active hinges
- Returns: `tuple[BodyV1, tuple[ActiveHinge, ActiveHinge, ActiveHinge, ActiveHinge]]`

#### `gecko_v2()` function
- Creates a more complex "gecko-like" robot body with 6 active hinges
- Returns: `tuple[BodyV2, tuple[ActiveHinge, ActiveHinge, ActiveHinge, ActiveHinge, ActiveHinge, ActiveHinge]]`

These functions define the physical structure of the robot, including the arrangement of hinges and bricks.

### 3. Brain Creation

- Uses `BrainCpgNetworkStatic` to create a CPG network brain
- The brain's weights are provided as an input parameter

### 4. Remote Control Configuration

- `Config` class is used to set up the remote control parameters
- Includes mapping of active hinges to GPIO pins
- Sets run duration, control frequency, and initial hinge positions

### 5. Main Control Function: `main(weights, bound)`

#### Parameters:
- `weights`: List of 9 float values representing CPG network weights
- `bound`: Float value limiting the maximum absolute value of weights

#### Key steps:
1. Adjusts weights to stay within the specified bound
2. Creates the robot body and brain
3. Sets up hinge mapping to GPIO pins
4. Configures the robot control parameters
5. Initiates video recording in a separate process
6. Runs the robot using `run_remote()`
7. Stops video recording
8. Returns the path of the recorded video

### 6. Video Recording

#### `record(stop_event, time_stamp)` function:
- Parameters:
  - `stop_event`: A multiprocessing Event to signal when to stop recording
  - `time_stamp`: String timestamp for naming the output video file
- Uses OpenCV to record video from an IP camera
- Runs in a separate process to allow concurrent operation with robot control

### 7. Movement Analysis

#### `compute_distance_on_videos(args, video_path)` function:
- Parameters:
  - `args`: An object containing various configuration parameters
  - `video_path`: String path to the recorded video file
- Uses the DREAM network for keypoint detection in the recorded video
- Analyzes the recorded video using the DREAM network
- Detects keypoints (head, box, joints) in each frame
- Calculates the distance traveled by the robot based on head keypoint movement

### 8. Optimization Process

The script implements a simple optimization loop:

1. Initialization:
   - Loads previous progress if available, otherwise starts with random weights
   - Sets up logging lists for weights, distances, and video paths

2. Iteration (80 times by default):
   - Generates 5 sets of weights by adding Gaussian noise to the best weights so far
   - Runs the robot with each set of weights
   - Analyzes the movement for each run
   - Selects the weights that resulted in the longest distance traveled
   - Updates the best weights if a new best is found
   - Logs the results and saves progress

3. Data Saving:
   - Saves all logged data (weights, distances, video paths) to a JSON file after each iteration

### 9. Configuration and Utility Classes

#### `Arguments` class:
Defines paths and configuration options for the DREAM network and output saving.

#### `NumpyEncoder` class:
A custom JSON encoder to handle NumPy types when saving data.

### 10. Key Parameters and Their Impacts

- `scale` (0.01): Controls the magnitude of noise added to weights during optimization
- `bound` (2.0): Limits the maximum absolute value of weights
- `iter` (80): Number of optimization iterations to perform

These parameters significantly affect the exploration-exploitation balance in the optimization process.

## Usage

1. Ensure all dependencies are installed (OpenCV, PyTorch, DREAM network, etc.)
2. Set up the physical robot and ensure it's connected to the network
3. Adjust the `Arguments` class with appropriate paths and settings
4. Run the script

## Notes and Customization

- The script is designed for a specific robot configuration and may need adjustments for different setups
- It assumes the availability of a pre-trained DREAM network for keypoint detection
- The optimization process is a simple exploration and might benefit from more sophisticated algorithms for better results
- The robot body can be easily modified by changing the `gecko_v2()` function
- The optimization process can be enhanced by implementing more sophisticated algorithms
- The DREAM network configuration and video analysis can be adjusted for different robot designs or environments

This comprehensive guide provides a detailed understanding of the code structure, key functions, and the overall workflow. It highlights the interplay between robot control, video recording, movement analysis, and the optimization process. Users can follow this guide to understand the code's operation and make necessary adjustments for their specific use cases.