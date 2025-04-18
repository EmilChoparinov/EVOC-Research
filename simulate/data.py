from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.modular_robot.body.v2 import BodyV2
from functools import reduce

import pandas as pd
import numpy as np
import ast
import simulate.stypes as stypes
import simulate.ea as ea
import pandas as pd
import cv2
import copy

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /             SELECTION / QUERIES              \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
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

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /             DATA UTILITIES                   \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
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

def inverse_lerp(a, low, high):
    assert(high - low != 0)
    return (a - low) / (high - low)

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /             HEURISTICS                       \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def compute_scaling_factor(df_robot: pd.DataFrame, df_animal: pd.DataFrame):
    # The scaling factor is computed in order to scale the robot to be of 
    # similar magnititude to the animal. This is how the heuristic works:
    #
    # 1. Select point cloud R from the first frame. This must be always 0 to 
    #    keep it deterministic
    # 2. Select some point cloud reference A from an animal frame. This can be
    #    any frame because the animal data is fixed.
    # 3. Calculate the mean of all points R and A and subtract the respective 
    #    positions in the point cloud to convert all coordinates to be relative
    # 4. Quantify the variance of both point clouds and take the ratio between
    #    them. This is the scale factor
    # 5. We take the square root because variance will give us the square of the
    #    scaling factor, because the squared *_center^2 within the ratio. We
    #    take the square root to recover the linear scaling factor
    #
    # SCALE FACTOR ALGORITHM LIMITATIONS:
    # 1. This algorithm essentially assumes that scaling is uniform across all
    #    dimensions which it is obviously not for the robot. We would need to do
    #    a more complex approach to capture the true distorted scale factor.
    #
    # 2. This algorithm assumes the clouds are already aligned in orientation. I
    #    believe the animal starts with a slight tilt relative to the x-axis. We
    #    could try to boost match-ability by ensuring the body vectors 
    #    (rear -> head) of both animals are aligned before computing the scale
    #    factor
    #
    # The reasoning to use this over a linear computation is because this will be
    # more robust to noise. Even though the point clouds from the robot are clean
    # I know the animal data has had some problems. So this is probably the best
    # approach even though there is low variance between the points and the 
    # center. The distribution should be somewhat uniform.
    NR_OF_FRAMES = 30
    robot_frames = [df_robot.iloc[i] for i in range (NR_OF_FRAMES)]
    animal_frames = [df_animal.iloc[i] for i in range (NR_OF_FRAMES)]

    S = 0
    for i in range (NR_OF_FRAMES):
        R = np.array([list(robot_frames[i][point]) for point in point_definition])
        A = np.array([list(animal_frames[i][point]) for point in point_definition])

        R_center = R - R.mean(axis=0)
        A_center = A - A.mean(axis=0)

        S += np.sqrt(np.sum(A_center**2) / np.sum(R_center**2))
    
    return S / NR_OF_FRAMES

def apply_front_limb_heuristic(df: pd.DataFrame):
    # This is a bandaid for the robot we use literally not matching the animal.
    # The front limbs of the animal are in different length. So we scale the
    # points
    factor = 15/19
    for limb in front_limb_group:
        df[limb] = df[limb].apply(lambda x: (x[0] * factor, x[1] * factor))
    return df

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


# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /             TRANSFORMS                       \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
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

def apply_statistics(df_behavior: pd.DataFrame, scores: dict[str, list[float]], state: stypes.EAState, gen_i: int):
    df_behavior['generation'] = gen_i
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
    
    for type, score in scores.items():
        df_behavior[type] = score

def create_video_state(state: stypes.EAState):
    frame_height = 720
    frame_width = 1080
    center_xy = (frame_width // 2, frame_height // 2)
    
    last_gen = state.generation - 1

    robot_behavior = convert_tuple_columns(pd.read_csv(
        ea.file_idempotent(state)).query(f"generation == {last_gen}"))
    animal_behavior = copy.deepcopy(state.animal_data)

    # Snap both to center across all rows
    def snap_to_center(row):
        center = row["middle"]
        for point in point_definition:
            row[point] = (row[point][0] - center[0], row[point][1] - center[1])
        return row

    robot_behavior = robot_behavior.apply(snap_to_center, axis=1)
    animal_behavior = animal_behavior.apply(snap_to_center, axis=1)

    robot_color = (0, 0, 255)
    animal_color = (0, 0, 0)

    video = cv2.VideoWriter(
        f"{ea.file_idempotent(state)}.mp4",
        cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height)) 

    frame_count = min(len(robot_behavior), len(animal_behavior))
    robot_behavior = robot_behavior.iloc[:frame_count]
    animal_behavior = animal_behavior.iloc[:frame_count]

    for frame in range(frame_count):
        frame_robot = robot_behavior.iloc[frame]
        frame_animal = animal_behavior.iloc[frame]

        # Create white background - fixed dimensions
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # Add description of video
        cv2.putText(img, 
                    f"{state.similarity_type} @ Alpha {state.alpha}",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0,0,0), 2, cv2.LINE_AA)
        
        cv2.putText(img, 
                    f"{frame}/{frame_count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0,0,0), 2, cv2.LINE_AA)

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
