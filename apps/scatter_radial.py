import pandas as pd

# Load the CSV file
file_path = 'most-fit-xy-run-1.csv'  # Path to the file
data = pd.read_csv(file_path)

# Convert stringified tuples to actual tuples in the DataFrame
def convert_to_tuple(string):
    return tuple(map(float, string.strip("()").split(",")))

# Select relevant columns to adjust body part positions
body_parts = ['head', 'middle', 'rear', 'right_front', 'left_front', 'right_hind', 'left_hind']
for part in body_parts:
    data[part] = data[part].apply(convert_to_tuple)

# Filter data for generation_id 290
generation_290_data = data[data['generation_id'] == 0].copy()

# Define the function to adjust positions
def fix_torso_position(data: pd.DataFrame, torso_part: str = 'middle') -> pd.DataFrame:
    """
    Adjust the positions of all body parts so that the torso (middle) is fixed at (0,0).

    Parameters:
    - data: pd.DataFrame containing body part positions. Each column is a part (e.g., 'middle', 'rear', etc.),
            and each value is a tuple (x, y).
    - torso_part: The name of the part to be fixed at (0,0) (default is 'middle').

    Returns:
    - pd.DataFrame with adjusted positions for all parts.
    """
    adjusted_data = data.copy()

    for index, row in data.iterrows():
        # Extract the (x, y) position of the torso
        torso_position = row[torso_part]
        if not isinstance(torso_position, (tuple, list)) or len(torso_position) != 2:
            raise ValueError(f"Invalid torso position at row {index}: {torso_position}")
        torso_x, torso_y = torso_position

        # Adjust all parts relative to the torso
        for part in data.columns:
            if part in ['generation_id', 'generation_best_fitness_score', 'frame_id', 'center-euclidian', 'alpha', 'fitness_function']:
                continue  # Skip non-position columns
            part_x, part_y = row[part]
            adjusted_data.at[index, part] = (part_x, part_y)

    return adjusted_data

# Adjust positions for generation 290
adjusted_data_290 = fix_torso_position(generation_290_data, torso_part='middle')

print(adjusted_data_290.head(100))

# Ensure the 'middle' column contains tuples (x, y) after adjustment
if 'middle' in adjusted_data_290.columns:
    # Extract the x and y components separately
    middle_x = adjusted_data_290['middle'].apply(lambda pos: pos[0])
    middle_y = adjusted_data_290['middle'].apply(lambda pos: pos[1])

    # Calculate min and max for both x and y
    middle_min_x, middle_max_x = middle_x.min(), middle_x.max()
    middle_min_y, middle_max_y = middle_y.min(), middle_y.max()

    print("Middle Column Min and Max:")
    print(f"Min X: {middle_min_x}, Max X: {middle_max_x}")
    print(f"Min Y: {middle_min_y}, Max Y: {middle_max_y}")

# Display the first 100 rows of the adjusted DataFrame
print(adjusted_data_290.head(100))

import matplotlib.pyplot as plt


# Function to plot and save the first 100 rows of the adjusted DataFrame with different colors for each keypoint
def plot_and_save_first_100_rows_with_colors(data, body_parts, colors, save_path="keypoints_plot.png"):
    """
    Plots the keypoints of the first 100 rows in the DataFrame with unique colors for each keypoint
    and saves the plot to a file.

    Parameters:
    - data: The DataFrame containing the adjusted positions.
    - body_parts: List of columns representing body parts (e.g., 'head', 'middle').
    - colors: List of colors corresponding to each body part.
    - save_path: The file path to save the plot (default: 'keypoints_plot.png').
    """
    plt.figure(figsize=(10, 10))

    for index, row in data.head(1).iterrows():
        for i, part in enumerate(body_parts):
            if isinstance(row[part], tuple) and len(row[part]) == 2:
                x, y = row[part]
                plt.scatter(x, y, color=colors[i], label=f"{part}" if index == 0 else "", alpha=0.6)

    plt.title("Points of Interest Scatter")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='small', ncol=2)
    plt.grid(True)
    plt.axis('equal')

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


# Define body parts and their corresponding colors
body_parts = ['head', 'middle', 'rear', 'right_front', 'left_front', 'right_hind', 'left_hind']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'purple']

# Call the plotting function and save the plot
plot_and_save_first_100_rows_with_colors(adjusted_data_290, body_parts, colors, save_path="first_100_keypoints.png")
