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

def most_fit(scores: npt.NDArray[np.float_], 
             df_behaviors: list[pd.DataFrame]):
    best_score_idx, best_score = max(enumerate(scores), key=lambda x: x[1])
    return (best_score, df_behaviors[best_score_idx])


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

def evaluate_by_angle(behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    # TODO: Move the angle information into tuples into the data section alongside
    #       everything else similar to this
    def calculate_angle_difference(frame):
        robot_angle1 = calculate_angle(frame["right_front"], 
                                       frame["left_hind"], 
                                       frame["left_front"])
        robot_angle2 = calculate_angle(frame["left_front"], 
                                       frame["right_hind"], 
                                       frame["right_front"])

        animal_angle1 = calculate_angle(animal_data.loc[frame.name, "right_front"],
                                       animal_data.loc[frame.name, "left_hind"],
                                       animal_data.loc[frame.name, "left_front"])
        animal_angle2 = calculate_angle(animal_data.loc[frame.name, "left_front"], 
                                       animal_data.loc[frame.name, "right_hind"], 
                                       animal_data.loc[frame.name, "right_front"])

        diff1 = abs(robot_angle1 - animal_angle1)
        diff2 = abs(robot_angle2 - animal_angle2)

        return (diff1 + diff2) / 2

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


def evaluate(behaviors: list[pd.DataFrame],state: stypes.EAState):
    distances = np.array([evaluate_by_distance(behavior) 
                                for behavior in behaviors])

    alpha = state.alpha

    match state.similarity_type:
        case "DTW":
            return [mix_ab(distance, 
                           data.value_rebound(
                               evaluate_by_dtw(behavior, state.animal_data),
                               (0, 100_000), (0, 2.5)),
                               alpha) 
                    for behavior, distance in zip(behaviors, distances)]            
        case "MSE":
            return [mix_ab(distance,
                           data.value_rebound(
                               evaluate_by_mse(behavior, state.animal_data),
                               (0, 30_000), (0, 2.5)),
                               alpha)
                    for behavior, distance in zip(behaviors, distances)]
        case "Angles":
            return [mix_ab(1 + distance / 500,
                           evaluate_by_angle(behavior, state.animal_data) / 360,
                            # evaluate_by_angle_dtw(behavior, state.animal_data),
                           alpha)
                    for behavior, distance in zip(behaviors, distances)]
        case "All_Angles":
            return [mix_ab(1 + distance / 500,
                           evaluate_all_angles(behavior, state.animal_data) / 360,
                            # evaluate_by_angle_dtw(behavior, state.animal_data),
                           alpha)
                    for behavior, distance in zip(behaviors, distances)]
        case _:
            raise NotImplementedError(
                f"The evaluator: `{state.similarity_type}` is not supported.")
