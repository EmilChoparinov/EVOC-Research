from revolve2.modular_robot import ModularRobot
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
def mix_ab(a: float, b:float, alpha: float) -> float:
    return alpha * a + (1 - alpha) * b

def most_fit(scores: npt.NDArray[np.float_], 
             df_behaviors: list[pd.DataFrame]):
    best_score_idx, best_score = max(enumerate(scores), key=lambda x: x[1])
    return (best_score, df_behaviors[best_score_idx])

def evaluate_by_distance(behavior: pd.DataFrame) -> float:
    return behavior.iloc[0]['head'][0] - behavior.iloc[-1]['head'][0]

def evaluate_by_mse(behavior: pd.DataFrame, animal: pd.DataFrame):
    # Apply MSE to column vector pairs, then take the average of all these pairs
    def apply_mse(frame):
        robot_frame = [frame[point] for point in data.point_definition]
        animal_frame = [animal.loc[frame.name, point] for point in data.point_definition]
        return mean_squared_error(robot_frame, animal_frame)
    behavior["MSE"] = behavior.apply(apply_mse,axis=1)
    
    # Apply mean to the column
    return np.mean(behavior["MSE"])

def evaluate_by_dtw(behavior: pd.DataFrame, animal: pd.DataFrame):
    # fastdtw library requires the points to be in a tuple of time series
    def serialize(frame):
        return [coord for point in 
                [frame[point] for point in data.point_definition] for coord in point]

    robot_timeseries = behavior.apply(serialize, axis=1).to_list()
    animal_timeseries = animal.apply(serialize, axis=1).to_list()

    # fastdtw library requires the time series to be of the same length
    normal_len = min(len(robot_timeseries), len(animal_timeseries))
    robot_timeseries = robot_timeseries[:normal_len]
    animal_timeseries = animal_timeseries[:normal_len]

    distance, _ = fastdtw(
        robot_timeseries, animal_timeseries, dist=euclidean)

    return distance

def calculate_angle(p1,p2,p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    vec1 = p2 - p1
    vec2 = p3 - p1

    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical instability
    return np.degrees(angle)

def evaluate_by_angle(behavior: pd.DataFrame, animal_data: pd.DataFrame) -> float:
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

    return np.mean(behavior["Angle_Diff"])  # Return overall mean difference

def evaluate(behaviors: list[pd.DataFrame],state: stypes.EAState):
    distances = np.array([evaluate_by_distance(behavior) 
                                for behavior in behaviors])

    if state.similarity_type == "DTW":
        return [mix_ab(
                    distance, 
                    data.value_rebound(
                        evaluate_by_dtw(behavior, state.animal_data),
                        (0, 100_000), (0, 2.5)) ,
                    state.alpha) 
                for behavior, distance in zip(behaviors, distances)]
    
    if state.similarity_type == "MSE":
        return [mix_ab(
                    distance,
                    data.value_rebound(
                        evaluate_by_mse(behavior, state.animal_data),
                        (0, 30_000), (0, 2.5)), 
                    state.alpha) 
                for behavior, distance in zip(behaviors, distances)]

    if state.similarity_type == "Angles":
        return [mix_ab(
                    distance, 
                    evaluate_by_angle(behavior, state.animal_data),
                    state.alpha) 
                for behavior, distance in zip(behaviors, distances)]