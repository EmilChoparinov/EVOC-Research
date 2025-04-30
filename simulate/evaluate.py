import math

from revolve2.modular_robot import ModularRobot
import logging
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import simulate.data as data
import ast
import pandas as pd
import numpy as np
import numpy.typing as npt
from revolve2.modular_robot_simulation import ModularRobotSimulationState
import simulate.stypes as stypes
from sklearn.metrics import mean_squared_error

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /           EVAL UTILITIES                     \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def mix_ab(a: float, b:float, alpha: float) -> float:
    return alpha * a + (1 - alpha) * b

def most_fit(scores: dict[str, npt.NDArray[np.float_]], 
             df_behaviors: list[pd.DataFrame]):
    ab_scores = scores['data_ab']
    best_score_idx, best_score = max(enumerate(ab_scores), key=lambda x: x[1])

    return ({data_line: evals[best_score_idx] 
             for data_line,evals in scores.items()}, df_behaviors[best_score_idx])

def calculate_angle(p1, p2, p3):
    P = np.array([p1, p2, p3])

    vec1 = P[0] - P[1]
    vec2 = P[2] - P[1]

    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])

    angle = np.degrees(abs(angle2 - angle1))

    return angle

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /               EVALUATORS                     \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def evaluate_by_distance(behavior: pd.DataFrame) -> float:
    return -max(behavior.iloc[-1]['head'][1] - behavior.iloc[0]['head'][1], 0)


def evaluate_nr_of_bad_frames(behavior: pd.DataFrame) -> float:
    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0]) **2 + (point1[1] - point2[1]) **2)

    nr = [0] * 17
    limbs = ['right_front', 'left_front', 'right_hind', 'left_hind']

    nr_touch = [0] * 4 # The amount of consecutive touches
    nr_air = [0] * 4 # The amount of consecutive air

    for i in range(1, len(behavior)):
        for j, limb in enumerate(limbs):
            limb_touch = (behavior.iloc[i][limb][2] < 0.044)
            if limb_touch:
                nr_touch[j] += 1
                nr_air[j] = 0
            else:
                nr_touch[j] = 0
                nr_air[j] += 1

            if nr_touch[j] >= 120: # If a limb is stuck on the ground for 4 sec or more.
                nr[j] += 1
            if nr_air[j] >= 120: # If a limb is stuck in the air for 4 sec or more.
                nr[j + 4] += 1

        if min(nr_touch[0], nr_touch[1]) >= 5: # If both front limbs are stuck on the ground for 1/6 sec or more.
            nr[8] += 1
        if min(nr_touch[2], nr_touch[3]) >= 5: # If both back limbs are stuck on the ground for 1/6 sec or more.
            nr[9] += 1
        if min(nr_air[0], nr_air[1]) >= 60: # If both front limbs are stuck in the air for 2 sec or more.
            nr[10] += 1
        if min(nr_air[2], nr_air[3]) >= 60: # If both back limbs are stuck in the air for 2 sec or more.
            nr[11] += 1

        dist_right = dist(behavior.iloc[i]["right_front"], behavior.iloc[i]["right_hind"])
        dist_left = dist(behavior.iloc[i]["left_front"], behavior.iloc[i]["left_hind"])
        if dist_right < 34: # If the right limbs are too close.
            nr[12] += 1
            if dist_right < 24:
                nr[12] += 1
        if dist_left < 34: # If the left limbs are too close.
            nr[13] += 1
            if dist_left < 24:
                nr[13] += 1

        if behavior.iloc[i]['head'][2] < 0.0748: # If the head is very close to the ground.
            nr[14] += 1
        #if nr_touch[0] != 0 and nr_touch[3] != 0 and nr_touch[1] == 0 and nr_touch[2] == 0: # Left step
        #    nr[15] -= 1
        #if nr_touch[1] != 0 and nr_touch[2] != 0 and nr_touch[0] == 0 and nr_touch[3] == 0: # Right step
        #    nr[16] -= 1


    print(nr)
    print(sum(nr))
    return sum(nr)


def evaluate_by_mse(behavior: pd.DataFrame, animal: pd.DataFrame):
    # Apply MSE to column vector pairs, then take the average of all these pairs
    def apply_mse(frame):
        robot_frame = np.array(
            [frame[point] for point in data.point_definition])
        animal_frame = np.array(
            [animal.loc[frame.name, point] for point in data.point_definition])
        return mean_squared_error(robot_frame, animal_frame)
    
    behavior["MSE"] = behavior.apply(apply_mse,axis=1)

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

def evaluate_by_2_angles(behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    # TODO: Move the angle information into tuples into the data section alongside
    #       everything else similar to this
    # TODO: Make MSE version using angle information
    def calculate_angle_difference(frame):
        robot_r_f = frame["right_front"]
        robot_r_h = frame["right_hind"]
        robot_l_f = frame["left_front"]
        robot_l_h = frame["left_hind"]

        animal_r_f = animal_data.loc[frame.name, "right_front"]
        animal_r_h = animal_data.loc[frame.name, "right_hind"]
        animal_l_f = animal_data.loc[frame.name, "left_front"]
        animal_l_h = animal_data.loc[frame.name, "left_hind"]


        robot_angle1 = calculate_angle(robot_r_f, robot_l_h, robot_l_f)
        animal_angle1 = calculate_angle(animal_r_f, animal_l_h, animal_l_f)

        robot_angle2 = calculate_angle(robot_l_f, robot_r_h, robot_r_f)
        animal_angle2 = calculate_angle(animal_l_f, animal_r_h, animal_r_f)

        diff1 = abs(robot_angle1 - animal_angle1)
        diff2 = abs(robot_angle2 - animal_angle2)

        return (diff1 + diff2) / 2

    behavior["Angle_Diff"] = behavior.apply(calculate_angle_difference, axis=1)

    return np.mean(behavior["Angle_Diff"])

import numpy as np
import pandas as pd

def evaluate_mechanical_work(behavior: pd.DataFrame, animal_data: pd.DataFrame):
    targets = data.point_definition  # List of joint names
    target_mass = data.point_mass    # Mass per joint
    
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

def evaluate_by_4_angles(behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    def calculate_angle_difference(frame):
        robot_r_f = frame["right_front"]
        robot_r_h = frame["right_hind"]
        robot_l_f = frame["left_front"]
        robot_l_h = frame["left_hind"]

        animal_r_f = animal_data.loc[frame.name, "right_front"]
        animal_r_h = animal_data.loc[frame.name, "right_hind"]
        animal_l_f = animal_data.loc[frame.name, "left_front"]
        animal_l_h = animal_data.loc[frame.name, "left_hind"]

        robot_angles = []
        animal_angles = []

        robot_angles.append(calculate_angle(robot_r_f, robot_l_h, robot_l_f))
        animal_angles.append(calculate_angle(animal_r_f, animal_l_h, animal_l_f))

        robot_angles.append(calculate_angle(robot_l_f, robot_r_h, robot_r_f))
        animal_angles.append(calculate_angle(animal_l_f, animal_r_h, animal_r_f))

        robot_angles.append(calculate_angle(robot_r_h, robot_l_f, robot_l_h))
        animal_angles.append(calculate_angle(animal_r_h, animal_l_f, animal_l_h))

        robot_angles.append(calculate_angle(robot_l_h, robot_r_f, robot_r_h))
        animal_angles.append(calculate_angle(animal_l_h, animal_r_f, animal_r_h))

        s = 0; N = 4
        for i in range(N):
            s += abs(robot_angles[i] - animal_angles[i])
            # TODO: Try MSE here

        return s / N

    behavior = data.drop_z_axis(behavior)
    behavior["Angle_Diff"] = behavior.apply(calculate_angle_difference, axis=1)

    return np.mean(behavior["Angle_Diff"])


def evaluate_all_angles(behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    def calculate_angles_difference(frame):
        robot_angles = []
        animal_angles = []
        for i in range(5):
            for j in range(i + 1, 6):
                for k in range(j + 1, 7):
                    robot_angles.append(calculate_angle(frame[data.point_definition[i]], frame[data.point_definition[j]],
                                                        frame[data.point_definition[k]]))
                    robot_angles.append(calculate_angle(frame[data.point_definition[i]], frame[data.point_definition[k]],
                                                        frame[data.point_definition[j]]))
                    robot_angles.append(calculate_angle(frame[data.point_definition[j]], frame[data.point_definition[i]],
                                                        frame[data.point_definition[k]]))

                    animal_angles.append(calculate_angle(animal_data.loc[frame.name, data.point_definition[i]],
                                                         animal_data.loc[frame.name, data.point_definition[j]],
                                                         animal_data.loc[frame.name, data.point_definition[k]]))
                    animal_angles.append(calculate_angle(animal_data.loc[frame.name, data.point_definition[i]],
                                                         animal_data.loc[frame.name, data.point_definition[k]],
                                                         animal_data.loc[frame.name, data.point_definition[j]]))
                    animal_angles.append(calculate_angle(animal_data.loc[frame.name, data.point_definition[j]],
                                                         animal_data.loc[frame.name, data.point_definition[i]],
                                                         animal_data.loc[frame.name, data.point_definition[k]]))
        s = 0
        for i in range(len(robot_angles)):
            s += abs(robot_angles[i] - animal_angles[i])

        return s / len(robot_angles)

    behavior["Angle_Diff"] = behavior.apply(calculate_angles_difference, axis=1)
    return np.mean(behavior["Angle_Diff"])

def evaluate_by_angle_dtw(behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
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


def evaluate(behaviors: list[pd.DataFrame],state: stypes.EAState, gen_i: int):
    distance_scores = np.array([evaluate_by_distance(behavior) 
                                for behavior in behaviors])

    # alpha = data.clamp(data.inverse_lerp(gen_i, 100, 200), 0, 0.75)
    def ab_mixer(xs,ys): return [mix_ab(x,y,state.alpha) for x,y in zip(xs, ys)]

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
            angle_scores = [evaluate_all_angles(behavior, state.animal_data)
                            for behavior in behaviors]
            return {'data_distance': distance_scores,
                    'data_ab': ab_mixer(
                                    [(d + 500)/500 for d in distance_scores], 
                                    [a/360 for a in angle_scores]),
                    'data_all_angles': angle_scores}
        
        case _:
            raise NotImplementedError(
                f"The evaluator: `{state.similarity_type}` is not supported.")
