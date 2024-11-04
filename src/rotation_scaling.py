import pandas as pd
import numpy as np
from revolve2.experimentation.logging import setup_logging

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic

from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

import logging
import math

from typedef import simulated_behavior, genotype
from typing import Tuple, List
import data_collection
import evaluate
import config

import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List



def get_data_with_forward_center(robots: List[ModularRobot], behaviors: List) -> pd.DataFrame:
    """
    Parameters:
        robots: List of ModularRobot objects.
        behaviors: List of behavior states corresponding to each robot.

    Returns:
        DataFrame containing forward center data for each robot and an exploded view of body part positions.
    """
    robot_coord_list = []

    for index, (robot, states) in enumerate(zip(robots, behaviors)):
        columns = ['head', 'middle', 'rear', 'right_front', 'left_front', 'right_hind', 'left_hind']

        # Iterate through each state (frame)
        for frame_id, state in enumerate(states):
            csv_map = config.body_to_csv_map(robot.body)
            pose_func = state.get_modular_robot_simulation_state(robot).get_module_absolute_pose

            # Initialize a dictionary to hold coordinates for each body part
            coord_dict = {'robot_index': index, 'frame_id': frame_id}

            for col in columns:
                abs_pose = pose_func(csv_map[col])
                # Store the x and y positions in the dictionary
                coord_dict[col + '_x'] = abs_pose.position.x
                coord_dict[col + '_y'] = abs_pose.position.y

            # Append the dictionary to the list
            robot_coord_list.append(coord_dict)

    # Create a DataFrame from the list of dictionaries
    pd_coord_list = pd.DataFrame(robot_coord_list)

    # Calculate center for each robot based on coordinates
    # Here, you may want to average over all body parts
    x_columns = [col + '_x' for col in columns]
    y_columns = [col + '_y' for col in columns]

    pd_coord_list['center_x'] = pd_coord_list[x_columns].mean(axis=1)
    pd_coord_list['center_y'] = pd_coord_list[y_columns].mean(axis=1)
    pd_coord_list["forward_x"] = pd_coord_list["head_x"] - pd_coord_list["center_x"]
    pd_coord_list["forward_y"] = pd_coord_list["head_x"] - pd_coord_list["center_x"]
    print(pd_coord_list.head(11))
    return pd_coord_list


def translation_rotation(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize a list to hold transformed points
    transformed_data = []
    # Extract relevant points directly from the DataFrame
    head_first = df['head_x'].iloc[0]
    head_first = df['head_y'].iloc[0]
    center_first =df['center_x'].iloc[0]
    center_first =df['center_y'].iloc[0]
    forward_first = df['forward_x'].iloc[0]
    forward_first = df['forward_y'].iloc[0]

    theta = (np.pi / 2) - np.arctan2(forward_first, forward_first)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    for index, row in df.iterrows():
        transformed_points = {}
        transformed_points['robot_index'] = row.get('robot_index', np.nan)
        transformed_points['frame_id'] = row.get('frame_id', np.nan)
        points_to_transform = {
            'head': np.array([row.get('head_x', np.nan), row.get('head_y', np.nan)]),
            'center': np.array([row.get('center_x', np.nan), row.get('center_y', np.nan)]),
            'forward': np.array([row.get('forward_x', np.nan), row.get('forward_y', np.nan)]),
            'middle': np.array([row.get('middle_x', np.nan), row.get('middle_y', np.nan)]),
            'rear': np.array([row.get('rear_x', np.nan), row.get('rear_y', np.nan)]),
            'right_front': np.array([row.get('right_front_x', np.nan), row.get('right_front_y', np.nan)]),
            'left_front': np.array([row.get('left_front_x', np.nan), row.get('left_front_y', np.nan)]),
            'right_hind': np.array([row.get('right_hind_x', np.nan), row.get('right_hind_y', np.nan)]),
            'left_hind': np.array([row.get('left_hind_x', np.nan), row.get('left_hind_y', np.nan)]),
        }

        # Translate and rotate each point
        for point_name, point_value in points_to_transform.items():
            try:
                if np.isnan(point_value).any():
                    print(f"Skipping {point_name} due to missing value.")
                    continue

                # Translate the point by subtracting the head position
                translated_point = point_value - np.array([head_first, head_first])
                # Rotate the point using the rotation matrix
                rotated_point = rotation_matrix @ translated_point
                transformed_points[point_name] = f"({rotated_point[0]:.2f}, {rotated_point[1]:.2f})"

            except Exception as e:
                print(f"Error parsing {point_name} with value {point_value}: {e}")

        # Append transformed points for the current row
        transformed_data.append(transformed_points)
    transformed_data=pd.DataFrame(transformed_data)
    print(transformed_data.head(20))
    return transformed_data
