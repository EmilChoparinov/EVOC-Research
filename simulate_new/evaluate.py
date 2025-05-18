import math
from itertools import permutations

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import simulate_new.data as data
from data import calculate_angle
import pandas as pd
import numpy as np
from simulate_new.data import mix_ab, ab_mixer
import simulate_new.stypes as stypes
from sklearn.metrics import mean_squared_error

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /               CALCULATORS                    \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def calculate_2_angles(movement: pd.DataFrame):
    def calculate_angles(frame):
        rf = frame["right_front"]
        rh = frame["right_hind"]
        lf = frame["left_front"]
        lh = frame["left_hind"]

        angles = np.array([calculate_angle(rf, lh, lf), calculate_angle(lf, rh, rf)])
        return angles

    angle_series = movement.apply(calculate_angles, axis=1)
    return np.vstack(angle_series.values)  # shape = (901, 2)

def calculate_4_angles(movement: pd.DataFrame):
    def calculate_angles(frame):
        rf = frame["right_front"]
        rh = frame["right_hind"]
        lf = frame["left_front"]
        lh = frame["left_hind"]

        angles = np.array([calculate_angle(rf, lh, lf), calculate_angle(lf, rh, rf),
                  calculate_angle(rh, lf, lh), calculate_angle(lh, rf, rh)])

        return angles

    angle_series = movement.apply(calculate_angles, axis=1)
    return np.vstack(angle_series.values)  # shape = (901, 4)

def calculate_all_angles(movement: pd.DataFrame):
    point_indices = list(range(7))
    point_combos = list(permutations(point_indices, 3))
    point_names = data.point_definition

    def calculate_angles(frame):
        return np.array([
            calculate_angle(frame[point_names[i]], frame[point_names[j]], frame[point_names[k]])
            for i, j, k in point_combos
        ])

    angle_series = movement.apply(calculate_angles, axis=1)
    return np.vstack(angle_series.values)  # shape = (901, 210)

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /               EVALUATORS                     \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def evaluate_by_distance(robot_behavior: pd.DataFrame) -> float:
    return -max(robot_behavior.iloc[-1]['head'][1] - robot_behavior.iloc[0]['head'][1], 0)

def evaluate_by_2_angles(robot_behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    robot_angles = np.array(calculate_2_angles(robot_behavior)) # .shape = (901, 2)
    animal_angles = np.array(calculate_2_angles(animal_data)) # .shape = (901, 2)

    simple_differences = np.abs(robot_angles - animal_angles) # .shape = (901, 2)
    mean_differences = np.mean(simple_differences) # .shape = (2)
    score = np.mean(mean_differences) # .shape = (1)
    return score

def evaluate_by_4_angles(robot_behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    robot_angles = calculate_4_angles(robot_behavior) # .shape = (901, 4)
    animal_angles = calculate_4_angles(animal_data) # .shape = (901, 4)

    simple_differences = np.abs(robot_angles - animal_angles) # .shape = (901, 4)
    mean_differences = np.mean(simple_differences) # .shape = (4)
    score = np.mean(mean_differences) # .shape = (1)
    return score

def evaluate_by_all_angles(robot_behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    robot_angles = np.array(calculate_all_angles(robot_behavior)) # .shape = (901, 210)
    animal_angles = np.array(calculate_all_angles(animal_data)) # .shape = (901, 210)

    simple_differences = np.abs(robot_angles - animal_angles) # .shape = (901, 210)
    mean_differences = np.mean(simple_differences) # .shape = (210)
    score = np.mean(mean_differences) # .shape = (1)
    return score

def evaluate_mechanical_work(behavior: pd.DataFrame):
    targets = data.point_definition  # List of joint names
    target_mass = data.point_mass  # Mass per joint

    dt = 30.0 / 900.0  # Time step duration

    def calc_work_joint(joint: str):
        joint_positions = np.vstack(behavior[joint].values)  # Shape: (time_steps, 3)
        mass = target_mass[joint]

        # displacement
        delta_x = np.diff(joint_positions, axis=0)

        # acceleration computed via central differences equation
        acceleration = (delta_x[1:] - delta_x[:-1]) / (dt ** 2)

        # paired against `acceleration`. For each acceleration[t], we pair it
        # with the displacement delta[t]
        delta_x_paired = delta_x[1:]

        # work computation
        work = mass * np.sum(acceleration * delta_x_paired, axis=1)

        return work

    # concatenate work from all joints and timesteps
    all_work = np.concatenate([calc_work_joint(joint) for joint in targets])

    # For total "effort" (absolute work), use:
    # The absolute work, used for training
    total_work = np.sum(np.abs(all_work))

    # The net work, used for testing
    # total_work = np.sum(all_work)

    return total_work

# ----------------------------------------------------------------------------------------------------------------------
def evaluate_by_2_angles_dtw(behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    min_len = min(len(behavior), len(animal_data))
    behavior = behavior[:min_len]
    animal_data = animal_data[:min_len]

    angle1 = behavior.apply(lambda r: calculate_angle(
        r["right_front"], r["left_hind"], r["left_front"]
    ) ,axis=1).values

    angle1_animal = animal_data.apply(lambda r: calculate_angle(
        r["right_front"], r["left_hind"], r["left_front"]
    ) ,axis=1).values

    angle2 = behavior.apply(lambda r: calculate_angle(
        r["left_front"], r["right_hind"], r["right_front"]
    ) ,axis=1).values

    angle2_animal = animal_data.apply(lambda r: calculate_angle(
        r["left_front"], r["right_hind"], r["right_front"]
    ) ,axis=1).values

    distance, _ = fastdtw(
         np.column_stack((angle1, angle2)), np.column_stack((angle1_animal, angle2_animal)))
    
    return distance

def evaluate_by_mse(behavior: pd.DataFrame, animal: pd.DataFrame):
    # Apply MSE to column vector pairs, then take the average of all these pairs
    def apply_mse(frame):
        robot_frame = np.array(
            [frame[point] for point in data.point_definition])
        animal_frame = np.array(
            [animal.loc[frame.name, point] for point in data.point_definition])
        return mean_squared_error(robot_frame, animal_frame)

    behavior["MSE"] = behavior.apply(apply_mse, axis=1)

    # Apply mean to the column
    return np.mean(behavior["MSE"])

def evaluate_by_dtw(behavior: pd.DataFrame, animal: pd.DataFrame):
    # fastdtw library requires the points to be in a tuple of time series
    def serialize(frame):
        return np.array([coord for point in
                         [frame[point] for point in data.point_definition]
                         for coord in point])

    robot_timeseries = behavior.apply(serialize, axis=1).to_list()
    animal_timeseries = animal.apply(serialize, axis=1).to_list()

    # fastdtw library requires the time series to be of the same length
    normal_len = min(len(robot_timeseries), len(animal_timeseries))
    robot_timeseries = robot_timeseries[:normal_len]
    animal_timeseries = animal_timeseries[:normal_len]

    distance, _ = fastdtw(
        robot_timeseries, animal_timeseries, dist=euclidean)

    return distance

def evaluate_nr_of_bad_frames(behavior: pd.DataFrame) -> float:
    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    nr = [0] * 17
    limbs = ['right_front', 'left_front', 'right_hind', 'left_hind']

    nr_touch = [0] * 4  # The amount of consecutive touches
    nr_air = [0] * 4  # The amount of consecutive air

    for i in range(1, len(behavior)):
        for j, limb in enumerate(limbs):
            limb_touch = (behavior.iloc[i][limb][2] < 0.044)
            if limb_touch:
                nr_touch[j] += 1
                nr_air[j] = 0
            else:
                nr_touch[j] = 0
                nr_air[j] += 1

            if nr_touch[j] >= 120:  # If a limb is stuck on the ground for 4 sec or more.
                nr[j] += 1
            if nr_air[j] >= 120:  # If a limb is stuck in the air for 4 sec or more.
                nr[j + 4] += 1

        if min(nr_touch[0], nr_touch[1]) >= 5:  # If both front limbs are stuck on the ground for 1/6 sec or more.
            nr[8] += 1
        if min(nr_touch[2], nr_touch[3]) >= 5:  # If both back limbs are stuck on the ground for 1/6 sec or more.
            nr[9] += 1
        if min(nr_air[0], nr_air[1]) >= 60:  # If both front limbs are stuck in the air for 2 sec or more.
            nr[10] += 1
        if min(nr_air[2], nr_air[3]) >= 60:  # If both back limbs are stuck in the air for 2 sec or more.
            nr[11] += 1

        dist_right = dist(behavior.iloc[i]["right_front"], behavior.iloc[i]["right_hind"])
        dist_left = dist(behavior.iloc[i]["left_front"], behavior.iloc[i]["left_hind"])
        if dist_right < 34:  # If the right limbs are too close.
            nr[12] += 1
            if dist_right < 24:
                nr[12] += 1
        if dist_left < 34:  # If the left limbs are too close.
            nr[13] += 1
            if dist_left < 24:
                nr[13] += 1

        if behavior.iloc[i]['head'][2] < 0.0748:  # If the head is very close to the ground.
            nr[14] += 1
        # if nr_touch[0] != 0 and nr_touch[3] != 0 and nr_touch[1] == 0 and nr_touch[2] == 0: # Left step
        #    nr[15] -= 1
        # if nr_touch[1] != 0 and nr_touch[2] != 0 and nr_touch[0] == 0 and nr_touch[3] == 0: # Right step
        #    nr[16] -= 1

    print(nr)
    print(sum(nr))
    return sum(nr)
# ----------------------------------------------------------------------------------------------------------------------

def evaluate(behaviors: list[pd.DataFrame],state: stypes.EAState, gen_i: int=-1):
    distance_scores = np.array([evaluate_by_distance(behavior) 
                                for behavior in behaviors])

    if gen_i != -1:
        alpha = data.clamp(data.inverse_lerp(gen_i, 100, 200), 0, 0.75)

    def ab_mixer(xs,ys): return [mix_ab(x,y,alpha) for x,y in zip(xs, ys)]

    match state.similarity_type:
        case "DTW":
            dtw_scores = [evaluate_by_dtw(behavior, state.animal_data)
                          for behavior in behaviors]
            
            return {'data_distance': distance_scores,
                    'data_ab': ab_mixer(distance_scores, dtw_scores),
                    'data_dtw': dtw_scores}
        
        case "MSE":
            return [mix_ab(distance,
                           data.value_rebound(
                               evaluate_by_mse(behavior, state.animal_data),
                               (0, 30_000), (0, 2.5)),
                               state.alpha)
                    for behavior, distance in zip(behaviors, distance_scores)]

        case "2_Angles":
            angle_scores = [evaluate_by_2_angles(behavior, state.animal_data)
                            for behavior in behaviors]
            return {'data_distance': [(d + 500)/500 for d in distance_scores],
                    'data_ab': ab_mixer(
                        [(d + 500)/500 for d in distance_scores],
                        [a/360 for a in angle_scores]),
                    'data_2_angles': angle_scores}

        case "4_Angles":
            angle_scores = [evaluate_by_4_angles(behavior, state.animal_data)
                          for behavior in behaviors]
            return {'data_distance': [(d + 500)/500 for d in distance_scores],
                    'data_ab': ab_mixer(
                        [(d + 500)/500 for d in distance_scores],
                        [a/360 for a in angle_scores]),
                    'data_4_angles': angle_scores}

        case "All_Angles":
            angle_scores = [evaluate_by_all_angles(behavior, state.animal_data)
                            for behavior in behaviors]
            return {'data_distance': distance_scores,
                    'data_ab': ab_mixer(
                                    [(d + 500)/500 for d in distance_scores], 
                                    [a/360 for a in angle_scores]),
                    'data_all_angles': angle_scores}
        
        case _:
            raise NotImplementedError(
                f"The evaluator: `{state.similarity_type}` is not supported.")
