from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2, BrickV2, CoreV2

import numpy as np
import ast
import simulate_new.stypes as stypes
import pandas as pd

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

ref = BrickV2(0.0)
point_mass = {
    "head": CoreV2._FRAME_MASS,
    # We omit the mass of the joints. I think for now its more accurate
    # if we just use the mass of the boxes
    "middle":ref.mass,
    "rear":ref.mass,
    "left_hind":ref.mass,
    "right_hind":ref.mass,
    "left_front":ref.mass,
    "right_front":ref.mass
}

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


def mix_ab(a: float, b:float, alpha: float) -> float:
    return alpha * a + (1 - alpha) * b

def ab_mixer(xs,ys, a): return [mix_ab(x,y,a) for x,y in zip(xs, ys)]


def calculate_angle(p1, p2, p3):
    P = np.array([p1[:2], p2[:2], p3[:2]])

    vec1 = P[0] - P[1]
    vec2 = P[2] - P[1]

    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])

    return np.degrees(abs(angle2 - angle1))

def calculate_angle_batch(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    p1_xy = p1[:, :2]
    p2_xy = p2[:, :2]
    p3_xy = p3[:, :2]

    vec1 = p1_xy - p2_xy
    vec2 = p3_xy - p2_xy

    angle1 = np.arctan2(vec1[:, 1], vec1[:, 0])
    angle2 = np.arctan2(vec2[:, 1], vec2[:, 0])

    return np.degrees(np.abs(angle2 - angle1))


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

def rotate_dataset_z(df: pd.DataFrame, theta: float):
    # Define the rotation matrix
    rad_theta = np.deg2rad(theta)
    rotation_matrix = np.array([
        [np.cos(rad_theta), -np.sin(rad_theta), 0],
        [np.sin(rad_theta), np.cos(rad_theta), 0],
        [0, 0, 1]
    ])

    center: tuple[float, float] = df['head'].iloc[0]

    for point_to_rotate in point_definition:
        # Subtract the origin to rotate by
        df[point_to_rotate] = df[point_to_rotate].apply(
            lambda x: tuple(x0 - c if i < 2 else x0 for i, (x0, c) in enumerate(zip(x, center)))
        )

        # Apply the 3D rotation matrix
        df[point_to_rotate] = df[point_to_rotate].apply(
            lambda x: tuple(rotation_matrix @ np.array([x[0], x[1], x[2]]).T)
        )

    return df


# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /             TRANSFORMS                       \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def behaviors_to_dataframes(
        robots: list[ModularRobot], behaviors: list[stypes.behavior], 
        state: stypes.EAState, z_axis=False):
    def to_df(z):
        # Extract the dataset generated from Revolve2 into a CSV in the same
        # format as animal_data
        robot: ModularRobot = z[0]
        behavior: stypes.behavior = z[1]
        body_map = csv_to_body(robot.body)

        def col_map(col: str, pose):
            abs_pose = pose(body_map[col])
            if not z_axis:
                return (abs_pose.position.x, abs_pose.position.y)
            else:
                return (abs_pose.position.x, abs_pose.position.y, abs_pose.position.z)

        arr = []
        for idx, frame in enumerate(behavior):
            pose_func = frame.get_modular_robot_simulation_state(robot).get_module_absolute_pose

            row = {col: col_map(col, pose_func) for col in point_definition}
            row["Frame"] = idx
            arr.append(row)

        df = pd.DataFrame(arr)

        if not z_axis:
            rotate_dataset(df, -90)
        else:
            rotate_dataset_z(df, -90)

        return df

    return list(map(to_df, zip(robots, behaviors)))
