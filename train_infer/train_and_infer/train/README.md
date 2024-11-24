# DREAM Network Training

This project implements a training pipeline for the DREAM (Deep Robot-to-camera Extrinsics for Articulated Manipulators) network, designed for keypoint detection in robotic manipulators.

## How to Run
```bash
python3 train_network.py -i ./label_data_formal -ar arch_configs/dream_vgg_q.yaml -m manip_configs/gecko.yaml -o out/0916/ -e 1 -b 32 

```

## Code Structure

The main training script is `train_network.py`, which contains the following key functions:

1. `train_network(args)`: The main function that orchestrates the entire training process.

2. `generate_belief_map_visualizations(...)`: A utility function for generating belief map visualizations.

Key aspects of the code:

- Utilizes PyTorch for network operations and training.
- Implements custom dataset handling with `ManipulatorNDDSDataset`.
- Supports resuming training from checkpoints.
- Implements data augmentation and preprocessing.
- Uses both training and validation phases in each epoch.
- Saves the best model based on validation loss.

## Usage

To train the network, use the `train_network.py` script with appropriate arguments.

## Parameters Explained

- `-i, --input-data-path`: Path to the training data directory.
- `-ar, --architecture-config`: Path to the network architecture configuration file.
- `-m, --manipulator-config-path`: Path to the manipulator configuration file.
- `-o, --output-dir`: Directory to save training results.
- `-e, --epochs`: Number of training epochs.
- `-b, --batch-size`: Batch size for training.

Additional important parameters:

- `-t, --training-data-fraction`: Fraction of data used for training (default: 0.8).
- `-f, --force-overwrite`: Force overwrite of existing results.
- `-z, --optimizer`: Optimizer type (default: adam).
- `-lr, --learning-rate`: Learning rate (default: 0.0001).
- `-not-a, --not-augment-data`: Disable data augmentation.
- `-w, --num-workers`: Number of data loading workers (default: 8).
- `-g, --gpu-ids`: Specify GPU IDs for training.
- `-s, --random-seed`: Set a specific random seed.
- `-v, --verbose`: Enable verbose output.
- `-r, --resume-training`: Resume training from a checkpoint.
- 
## Key Features

- Custom network architecture and manipulator configurations.
- Data augmentation for improved generalization.
- Checkpoint-based training resumption.
- Best model saving based on validation loss.
- Detailed training logs and statistics generation.

## Data Handling

The code uses `ManipulatorNDDSDataset` for data loading and preprocessing. It supports:

- Splitting data into training and validation sets.
- Image preprocessing and normalization.
- Belief map generation for keypoint detection.

## Training Process

1. Network initialization or loading from checkpoint.
2. Data loading and preprocessing.
3. Iterative training over specified epochs.
4. Alternating between training and validation phases.
5. Loss calculation and backpropagation.
6. Model saving and logging.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- ruamel.yaml
- tqdm

## Note

## Detailed Code Explanation

### Main Components

1. **train_network(args)**
   - This is the main function that orchestrates the entire training process.
   - It handles argument parsing, data loading, network initialization, and the training loop.

2. **generate_belief_map_visualizations(...)**
   - A utility function for creating visual representations of belief maps.
   - Useful for debugging and understanding the network's keypoint predictions.

### Key Classes and Functions

1. **ManipulatorNDDSDataset**
   - Custom dataset class for handling NDDS (NVIDIA Deep learning Dataset Synthesizer) data.
   - Responsible for loading images, keypoints, and generating belief maps.
   - Implements data augmentation when enabled.

2. **dream.create_network_from_config_data(network_config)**
   - Factory function that creates the DREAM network based on the provided configuration.
   - Allows for flexible network architecture defined in the config file.

3. **dream_network.enable_training() / dream_network.enable_evaluation()**
   - Switches the network between training and evaluation modes.
   - Affects behavior of certain layers (e.g., Dropout, BatchNorm).

### Training Process Flow

1. **Data Preparation**
   - Loads data using ManipulatorNDDSDataset.
   - Splits data into training and validation sets.
   - Creates DataLoader instances for efficient batching.

2. **Network Initialization**
   - Creates the network using the provided architecture config.
   - Loads weights if resuming training.

3. **Training Loop**
   - Iterates through specified number of epochs.
   - For each epoch:
     a. Training Phase: Forward pass, loss calculation, backpropagation.
     b. Validation Phase: Forward pass, loss calculation (no backpropagation).
     c. Logs training and validation losses.
     d. Saves the best model based on validation loss.

4. **Optimization**
   - Uses the specified optimizer (default: Adam).
   - Applies learning rate as specified in arguments.

5. **Checkpointing**
   - Saves model state at the end of each epoch.
   - Allows resuming training from the last checkpoint.

### Configuration Files

1. **Architecture Config (arch_configs/dream_vgg_q.yaml)**
   - Defines the network architecture.
   - Specifies layers, activation functions, and other architectural details.

2. **Manipulator Config (manip_configs/gecko.yaml)**
   - Defines the manipulator-specific parameters.
   - Specifies keypoints, their names, and other manipulator-specific details.

### Data Augmentation

- Implemented in the ManipulatorNDDSDataset class.
- Includes operations like random flipping, rotation, and color jittering.
- Can be disabled using the `--not-augment-data` flag.

### Logging and Visualization

- Training progress is logged to a pickle file for each epoch.
- Best model is saved separately.
- Implements TensorBoard logging for real-time training visualization (if enabled).

### GPU Utilization

- Supports multi-GPU training.
- GPU IDs can be specified using the `--gpu-ids` argument.

### Error Handling and Validation

- Implements various assertions to check for data consistency and correct parameter usage.
- Validates configuration files and data paths before starting training.

## Getting Started for New Team Members

1. Familiarize yourself with the configuration files (architecture and manipulator configs).
2. Understand the data format expected by ManipulatorNDDSDataset.
3. Review the main training loop in `train_network` function to understand the training flow.
4. Check the argument parser to see all available command-line options.
5. Start with a small dataset and few epochs to ensure everything is working correctly.
6. Use the verbose mode (`-v`) for detailed output during initial runs.

## Potential Areas for Improvement/Extension

- Implement additional data augmentation techniques.
- Add support for different loss functions.
- Extend the architecture to support different backbone networks.
- Implement online evaluation metrics during training.
- Add support for distributed training across multiple machines.