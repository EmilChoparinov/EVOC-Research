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

    angle = np.degrees(abs(angle2 - angle1))

    return angle

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /             HEURISTICS                       \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def apply_front_limb_heuristic(df: pd.DataFrame, z_axis=False):
    # This is a bandaid for the robot we use literally not matching the animal.
    # The front limbs of the animal are in different length. So we scale the
    # points
    factor = 15/19
    if not z_axis:
        for limb in front_limb_group:
            df[limb] = df[limb].apply(lambda x: (x[0] * factor, x[1] * factor))
    else:
        for limb in front_limb_group:
            df[limb] = df[limb].apply(lambda x: (x[0] * factor, x[1] * factor, x[2]))
    return df

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

        # Apply the normalization process to make it more in line with animal
        # data. First we compute the scaling difference on the first frame and
        # Scale the points up
        if not z_axis:
            scale_factor = compute_scaling_factor(df, state.animal_data)
            for point in point_definition:
                df[point] = df[point]\
                    .apply(lambda x: (x[0] * scale_factor, x[1] * scale_factor))
            rotate_dataset(df, -90)
        else:
            scale_factor = compute_scaling_factor(df, state.animal_data)
            for point in point_definition:
                df[point] = df[point]\
                    .apply(lambda x: (x[0] * scale_factor, x[1] * scale_factor, x[2]))
            rotate_dataset_z(df, -90)

        # The front limbs are misconfigured on the robot. We apply a heuristic
        # s.t. the front and back are the same scale
        #apply_front_limb_heuristic(df, z_axis=z_axis)

        return df

    return list(map(to_df, zip(robots, behaviors)))
