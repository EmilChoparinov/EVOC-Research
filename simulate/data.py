from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.modular_robot.body.v2 import BodyV2
import pandas as pd
from functools import reduce
import numpy as np
import os
import ast
import simulate.stypes as stypes
import simulate.ea as ea
import pandas as pd
import cv2
import copy

csv_columns = [
    "generation", "score", "similarity_type", "Frame",
    "head", "middle", "rear", 
    "left_hind", "right_hind", "left_front", "right_front",
    "center_euclidian"
]

point_definition = [
    "head", "middle", "rear",
    "left_hind", "right_hind", "left_front", "right_front"
]

front_limb_group = ["left_front", "right_front"]

edge_definition = [
    ('head', 'left_front'),
    ('head', 'middle'),
    ('middle','rear'), 
    ('head', 'right_front'), 
    ('rear', 'right_hind'), 
    ('rear', 'left_hind')
]

def csv_to_body(body: BodyV2):
    return  {
        "head": body.core_v2,
        "middle": body.core_v2.back_face.bottom.attachment,
        "rear": body.core_v2.back_face.bottom.attachment.front.attachment,
        "right_front": body.core_v2.right_face.bottom.attachment,
        "left_front": body.core_v2.left_face.bottom.attachment,
        "right_hind":body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment,
        "left_hind": body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment
    }

def convert_tuple_columns(df: pd.DataFrame):
    for point in point_definition:
        df[point] = df[point].apply(ast.literal_eval)
    return df

def value_rebound(
        value, start_bound: tuple[float, float], end_bound: tuple[float, float]):
    ratio = (value - start_bound[0]) / (start_bound[1] - start_bound[0])
    return end_bound[0] + ratio * (end_bound[1] - end_bound[0])


def behaviors_to_dataframes(
        robots: list[ModularRobot], behaviors: list[stypes.behavior], 
        state: stypes.EAState):
    def to_df(z):
        # Extract the dataset generated from Revolve2 into a CSV in the same
        # format as animal_data
        robot: ModularRobot = z[0] 
        behavior: stypes.behavior = z[1]
        body_map = csv_to_body(robot.body)

        def col_map(col: str, pose):
            abs_pose = pose(body_map[col])
            return (abs_pose.position.x, abs_pose.position.y)

        arr = []
        for idx, frame in enumerate(behavior):
            pose_func = frame.get_modular_robot_simulation_state(robot)\
                .get_module_absolute_pose
            
            row = {col: col_map(col, pose_func) for col in point_definition}
            row["Frame"] = idx
            arr.append(row)

        df = pd.DataFrame(arr)

        # Apply the normalization process to make it more in line with animal
        # data. First we compute the scaling difference on the first frame and
        # Scale the points up
        scale_factor = compute_scaling_factor(df, state.animal_data)
        for point in point_definition:
            df[point] = df[point]\
                .apply(lambda x: (x[0] * scale_factor, x[1] * scale_factor))
        
        # We also need to rotate the dataset so they walk the same direction
        rotate_dataset(df, -90)

        # The front limbs are misconfigured on the robot. We apply a heuristic 
        # s.t. the front and back are the same scale
        apply_front_limb_heuristic(df)
        return df

    return list(map(to_df, zip(robots, behaviors)))

def apply_statistics(df_behavior: pd.DataFrame, score: float, state: stypes.EAState, gen_i: int):
    df_behavior['generation'] = gen_i
    df_behavior['score'] = score
    df_behavior['similarity_type'] = state.similarity_type

    # Calculate the sum of all points into `center_euclidian`
    df_behavior['center_euclidian'] = df_behavior.apply(
        lambda row: reduce(
            lambda acc, point: (acc[0] + point[0], acc[1]+ point[1]),
            [row[point] for point in point_definition], (0.0,0.0)), 
        axis=1)
    
    # Divide and get the average point
    df_behavior['center_euclidian'] = df_behavior["center_euclidian"].\
        apply(lambda x: 
              (x[0]/ len(point_definition), x[1] / len(point_definition)))

def compute_scaling_factor(df_robot: pd.DataFrame, df_animal: pd.DataFrame):
    robot_0 = df_robot.iloc[0]
    animal_0 = df_animal.iloc[0]

    R = np.array([
        list(robot_0[point]) for point in point_definition])
    
    A = np.array([
        list(animal_0[point]) for point in point_definition])

    R_center = R - R.mean(axis=0)
    A_center = A - A.mean(axis=0)
    
    return np.sqrt(np.sum(A_center**2) / np.sum(R_center**2))

def rotate_dataset(df: pd.DataFrame, theta: float):
    # Define the rotation matrix
    rad_theta = np.deg2rad(theta)
    rotation_matrix = np.array([
        [np.cos(rad_theta), -np.sin(rad_theta)], 
        [np.sin(rad_theta), np.cos(rad_theta)]])
    
    # We rotate relative to the origin of the subject. We mark the head as the
    # origin
    center: tuple[float,float] = df['head'].iloc[0]

    for point_to_rotate in point_definition:
        # Subtract the origin to rotate by
        df[point_to_rotate] = df[point_to_rotate].\
            apply(lambda x: tuple(x0 - c for x0,c in zip(x, center)))

        # Rotate the column
        df[point_to_rotate] = df[point_to_rotate]\
            .apply(lambda x: tuple(rotation_matrix @ x))
    
    return df

def apply_front_limb_heuristic(df: pd.DataFrame):
    factor = 15/19
    for limb in front_limb_group:
        # Convert point to 
        df[limb] = df[limb].apply(lambda x: (x[0] * factor, x[1] * factor))
    return df

def create_video_state(state: stypes.EAState):
    frame_height = 720
    frame_width = 1080
    center_xy = (frame_width // 2, frame_height // 2)
    
    last_gen = state.generation - 1

    robot_behavior = convert_tuple_columns(pd.read_csv(
        ea.file_idempotent(state)).query(f"generation == {last_gen}"))
    animal_behavior = state.animal_data

    robot_color = (0, 0, 255)
    animal_color = (0, 0, 0)

    video = cv2.VideoWriter(
        f"{ea.file_idempotent(state)}.avi",
        cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame_width, frame_height)) 

    frame_count = min(len(robot_behavior), len(animal_behavior))
    robot_behavior = robot_behavior.iloc[:frame_count]
    animal_behavior = animal_behavior.iloc[:frame_count]

    for frame in range(frame_count):
        frame_robot = robot_behavior.iloc[frame]
        frame_animal = animal_behavior.iloc[frame]

        # Create white background - fixed dimensions
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # Add points
        for point in point_definition:
            # Add Robot Point
            coord = frame_robot[point]
            screen_coord = (
                int(center_xy[0] + coord[0]),
                int(center_xy[1] - coord[1])
            )
            cv2.circle(img, screen_coord, 3, robot_color, -1)

            # Add Animal Point
            coord = frame_animal[point]
            screen_coord = (
                int(center_xy[0] + coord[0]),
                int(center_xy[1] - coord[1])
            )
            cv2.circle(img, screen_coord, 3, animal_color, -1)

        # Add Edges
        for p1, p2 in edge_definition:
            # Add Robot Edge
            coord_p1 = frame_robot[p1]
            coord_p2 = frame_robot[p2]
            
            screen_coord_p1 = (
                int(center_xy[0] + coord_p1[0]),
                int(center_xy[1] - coord_p1[1])
            )
            screen_coord_p2 = (
                int(center_xy[0] + coord_p2[0]),
                int(center_xy[1] - coord_p2[1])
            )
            
            cv2.line(img, screen_coord_p1, screen_coord_p2, robot_color, 2)

            # Add Animal Edge
            coord_p1 = frame_animal[p1]
            coord_p2 = frame_animal[p2]
            
            screen_coord_p1 = (
                int(center_xy[0] + coord_p1[0]),
                int(center_xy[1] - coord_p1[1])
            )
            screen_coord_p2 = (
                int(center_xy[0] + coord_p2[0]),
                int(center_xy[1] - coord_p2[1])
            )
            
            cv2.line(img, screen_coord_p1, screen_coord_p2, animal_color, 2)

        video.write(img)

    video.release()
    cv2.destroyAllWindows()
