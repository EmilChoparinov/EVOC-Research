# Gecko Robot Project: Advanced Bio-Inspired Robotics Platform

## Project Overview

This project, located at `~/Thesis/code/gecko_code_with_tutorial`, represents a cutting-edge system for the development, control, and analysis of a gecko-inspired robot. It integrates advanced robotics, computer vision, machine learning, and data analysis to create a comprehensive platform for bio-inspired robotic research.
I already package the venv folder, you can also create a new venv and install the requirements.txt file.
## Project Structure


## Key Components and Technical Insights

### 1. Robot Control (`robot_control_and_exp/robot_control/`)

- **Physical Robot Remote Control** (`examples/physical_robot_remote/`)
  - `main.py`: Core script for robot control, likely implementing:
    - Real-time control loops
    - Sensor data integration
    - Inverse kinematics for gecko-like locomotion
  - `arch_configs/`: Crucial for defining the robot's physical structure and capabilities
  - `dream/`: Implementation of DREAM model, potentially using:
    - Deep learning frameworks (e.g., PyTorch, TensorFlow)
    - Custom layers for robotic state estimation

**Technical Insight**: The presence of multiple backup files (`main_backup.py`, `main2_backup.py`, etc.) suggests an iterative development process, possibly testing different control strategies or locomotion patterns.

### 2. Web Camera Interface (`robot_control_and_exp/web_camera/`)

- `web_camera.py`: Likely includes:
  - Multi-camera synchronization
  - High-speed image capture capabilities
  - Possible integration with ROS (Robot Operating System) for distributed sensing

**Technical Insight**: The separation of web camera functionality indicates a modular design approach, allowing for easy integration of different vision systems or upgrading to more advanced cameras in the future.

### 3. Data Processing (`train_and_infer/data_process/`)

- `label.py`: Sophisticated tool for data annotation, possibly featuring:
  - GUI for efficient keypoint marking
  - Semi-automated labeling assistance
- `split-frames.py`: Critical for creating training datasets, likely including:
  - Intelligent frame selection algorithms
  - Data augmentation techniques

**Technical Insight**: The presence of `label_data_formal/` suggests a rigorous approach to data curation, crucial for developing accurate machine learning models.

### 4. Model Training and Inference (`train_and_infer/infer/`)

- `dream/`: Advanced implementation of the DREAM model:
  - Likely utilizes state-of-the-art deep learning architectures (e.g., transformers, graph neural networks)
  - Custom loss functions for precise keypoint detection in robotic applications

**Technical Insight**: The separation of inference code suggests a deployment-ready setup, possibly with optimizations for real-time performance on embedded systems.

### 5. Data Analysis (`train_and_infer/save_data/video2csv/`)

- `video2csv.py` and `video2csv_folders.py`: Sophisticated data conversion tools:
  - Likely implement parallel processing for handling large datasets
  - May include feature extraction algorithms for advanced motion analysis

**Technical Insight**: The batch processing capability (`video2csv_folders.py`) indicates scalability for large-scale experiments and data collection campaigns.

## Advanced Features and Potential Applications

1. **Adaptive Locomotion**: The robot likely implements adaptive gait patterns, adjusting to different surfaces and inclines in real-time.

2. **Multi-Modal Sensing**: Integration of visual data with other sensor types (e.g., IMU, force sensors) for robust state estimation.

3. **Reinforcement Learning Integration**: Potential for implementing RL algorithms for continuous improvement of locomotion strategies.

4. **Biomimetic Material Integration**: Possible use of gecko-inspired adhesive materials for enhanced climbing capabilities.

5. **Swarm Robotics Potential**: The modular design could be extended to multi-robot systems for collaborative tasks.

## Contribution Guidelines

- Adhere to PEP 8 style guide for Python code
- Implement comprehensive unit tests for all new features
- Document all APIs and maintain up-to-date inline comments
- Use git flow workflow for feature development and releases

# Gecko Robot Project: Advanced Bio-Inspired Robotics Platform

## Comprehensive Workflow Guidelines

### 1. Data Processing and Model Development

#### 1.1 Data Collection and Processing
- Navigate to train_and_infer/data_process/
- Use split-frames.py to extract frames from initial video footage:
  python split-frames.py --input_video path/to/video.mp4 --output_dir path/to/frames/
- Use label.py to annotate keypoints on extracted frames:
  python label.py --image_dir path/to/frames/ --output_file keypoints.json

#### 1.2 Model Training
- Navigate to train_and_infer/train/ (Note: This directory is implied but not explicitly shown in the structure)
- Prepare your training script (e.g., train_dream.py) using the labeled data
- Execute the training:
python3 train_network.py -i ./label_data_formal -ar arch_configs/dream_vgg_q.yaml -m manip_configs/gecko.yaml -o out/0916/ -e 1 -b 32 

#### 1.3 Model Inference
- Move to train_and_infer/infer/dream/
- Use the trained model for inference on new data:
  python video2csv.py 

### 2. Robot Control and Experimentation

#### 2.1 Setup Robot Control
- Navigate to robot_control_and_exp/robot_control/examples/physical_robot_remote/
- Review and modify main.py to incorporate your trained model and desired control algorithms

#### 2.2 Prepare Web Camera Monitoring
- Go to robot_control_and_exp/web_camera/
- Configure web_camera.py for your specific camera setup:
  python web_camera.py path/to/experiment_footage/

#### 2.3 Run Experiment
- Start the web camera recording
- Execute the robot control script:
  python main.py --model_path path/to/dream_model.pth --experiment_duration 600
- Monitor the experiment in real-time and ensure data is being recorded properly

### 3. Data Analysis

#### 3.1 Video to CSV Conversion
- Navigate to train_and_infer/save_data/video2csv/
- Process the experiment footage:
  python video2csv_folders.py --input_dir path/to/experiment_footage/ --output_dir path/to/csv_data/

#### 3.2 Raw Data Analysis
- The resulting CSV files in path/to/csv_data/ contain raw data for further analysis
- Use these files for:
  - Trajectory analysis
  - Performance metrics calculation
  - Comparison with simulation results

### 4. Iterative Improvement

- Analyze the results from the experiment
- Identify areas for improvement in:
  - Model accuracy
  - Robot control algorithms
  - Data collection process
- Repeat the process from step 1, incorporating insights gained

## Best Practices

- Maintain a consistent directory structure for each experiment
- Use version control (git) to track changes in code and configurations
- Document each experiment thoroughly, including:
  - Date and time
  - Specific configurations used
  - Observations and anomalies
- Regularly backup raw data and processed results

## Troubleshooting

- If the robot behaves unexpectedly:
  1. Check all physical connections
  2. Verify sensor data input
  3. Review recent changes in control algorithms
- For issues with the DREAM model:
  1. Ensure input data format matches training data
  2. Check for any preprocessing inconsistencies
  3. Verify model loading and inference pipeline
- If data analysis yields unexpected results:
  1. Validate the raw data in CSV files
  2. Check for any data conversion errors in video2csv process
  3. Review analysis scripts for logical errors

