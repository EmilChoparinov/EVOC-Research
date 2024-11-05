import pandas as pd
import numpy as np
from revolve2.experimentation.logging import setup_logging

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic

from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

import logging
import math

import ast
from typedef import simulated_behavior, genotype
from typing import Tuple, List

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
    # print(pd_coord_list.head(11))
    return pd_coord_list


def translation_rotation(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize a list to hold transformed points
    transformed_data = []
    # Extract relevant points directly from the DataFrame
    head_first_x = df['head_x'].iloc[0]
    head_first_y = df['head_y'].iloc[0]
    forward_first_x = df['forward_x'].iloc[0]
    forward_first_y = df['forward_y'].iloc[0]

    theta = (np.pi / 2) - np.arctan2(forward_first_y, forward_first_x)

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
                translated_point = point_value - np.array([head_first_x, head_first_y])
                # Rotate the point using the rotation matrix
                rotated_point = rotation_matrix @ translated_point
                transformed_points[point_name] = f"({rotated_point[0]:.2f}, {rotated_point[1]:.2f})"

            except Exception as e:
                print(f"Error parsing {point_name} with value {point_value}: {e}")

        # Append transformed points for the current row
        transformed_data.append(transformed_points)
    transformed_data=pd.DataFrame(transformed_data)
    # print(transformed_data.head(20))
    return transformed_data

def fitness_scaling(fitnesses: npt.NDArray[np.float_]):
    min_f = np.min(fitnesses)
    max_f = np.max(fitnesses)
    scaled_fitness = (fitnesses - min_f) / (max_f - min_f)
    # print('fitnesses',fitnesses)
    # print('scaled_fitnesses',scaled_fitness)
    return scaled_fitness

def size_scaling(df: pd.DataFrame) -> pd.DataFrame:
    def parse_tuple_string(s):
        return ast.literal_eval(s) if pd.notna(s) else (None, None)

    def extract_keypoints(data_dict):
        keypoints = []

        if isinstance(data_dict, dict):
            for name, coord in data_dict.items():
                if coord != (None, None):
                    keypoints.append(coord)
        elif isinstance(data_dict, pd.DataFrame) or isinstance(data_dict, pd.Series):
            keypoints = data_dict[['X (relative)', 'Y (relative)']].values.flatten()

        return np.array(keypoints).flatten()  # 将二维坐标转为一维向量

    global_max_distance_animal = 168.13387523042465

    global_max_distances = {i: 0 for i in range(10)}  # Initialize max distances for each robot index from 0 to 9
    coordinates_2_list = []


    for index, frame in df.iterrows():
        coordinates_2 = {
            'head': parse_tuple_string(frame.get('head', None)),
            'middle': parse_tuple_string(frame.get('middle', None)),
            'rear': parse_tuple_string(frame.get('rear', None)),
            'right_front': parse_tuple_string(frame.get('right_front', None)),
            'left_front': parse_tuple_string(frame.get('left_front', None)),
            'right_hind': parse_tuple_string(frame.get('right_hind', None)),
            'left_hind': parse_tuple_string(frame.get('left_hind', None)),
        }

        if None in coordinates_2.values() or any(v is None for v in coordinates_2.values()):
            print(f'Frame {index}: Incomplete robot coordinates: {coordinates_2}')

        coordinates_2_list.append(coordinates_2)

        # Retrieve robot_index for the frame
        robot_index = frame.get('robot_index', np.nan)

        # Calculate the max distance for this robot index
        if not np.isnan(robot_index):
            robot_head = np.array(coordinates_2['head']).astype(float)
            robot_boxes = np.array([coordinates_2[box_name] for box_name in
                                    ['middle', 'rear', 'right_front', 'left_front', 'right_hind', 'left_hind']]).astype(
                float)
            robot_distances = np.linalg.norm(robot_boxes - robot_head, axis=1)
            max_distance_robot = np.max(robot_distances)

            # Update the max distance for this robot_index if this distance is larger
            robot_index = int(robot_index)
            if max_distance_robot > global_max_distances[robot_index]:
                global_max_distances[robot_index] = max_distance_robot

    print(f'Global maximum distances per robot_index: {global_max_distances}')


    scaled_robot_data = []
    first_frame_scaled_coordinates = None
    first_frame_robot_coordinates = None

    for i in range(len(coordinates_2_list)):
        robot_coordinates = extract_keypoints(coordinates_2_list[i])
        robot_head = coordinates_2_list[i]['head']

        robot_index = df.iloc[i].get('robot_index', np.nan)

        max_distance_robot = global_max_distances.get(int(robot_index), 1)  


        robot_coordinates = robot_coordinates.reshape(-1, 2)
        robot_head = np.array(robot_head).reshape(1, 2)
        scaled_robot_coordinates = (robot_coordinates - robot_head) * (
                global_max_distance_animal / max_distance_robot) + robot_head


        if i == 0:
            first_frame_scaled_coordinates = scaled_robot_coordinates.copy()
            first_frame_robot_coordinates = robot_coordinates.copy()
            scaled_robot_coordinates[0] = (0, 0)
        else:
            if first_frame_scaled_coordinates is None or first_frame_robot_coordinates is None:
                continue
            for j in range(robot_coordinates.shape[0]):
                scaled_robot_coordinates[j] = first_frame_scaled_coordinates[j] + (
                        robot_coordinates[j] - first_frame_robot_coordinates[j])

        scaled_robot_data.append({
            'Frame': i,
            'robot_index': robot_index,

            'head': tuple(scaled_robot_coordinates[0]),
            'middle': tuple(scaled_robot_coordinates[1]),
            'rear': tuple(scaled_robot_coordinates[2]),
            'right_front': tuple(scaled_robot_coordinates[3]),
            'left_front': tuple(scaled_robot_coordinates[4]),
            'right_hind': tuple(scaled_robot_coordinates[5]),
            'left_hind': tuple(scaled_robot_coordinates[6]),
        })

    scaled_robot_data=pd.DataFrame(scaled_robot_data)
    # print(scaled_robot_data.head(5))
    return scaled_robot_data
