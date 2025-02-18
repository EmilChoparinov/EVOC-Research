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

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /           EVAL UTILITIES                     \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def mix_ab(a: float, b:float, alpha: float) -> float:
    return alpha * a + (1 - alpha) * b

def most_fit(scores: npt.NDArray[np.float_], 
             df_behaviors: list[pd.DataFrame]):
    best_score_idx, best_score = max(enumerate(scores), key=lambda x: x[1])
    return (best_score, df_behaviors[best_score_idx])


def calculate_angle(p1,p2,p3):
    P = np.array([p1, p2, p3])
    
    vec1 = P[1] - P[0]
    vec2 = P[2] - P[0]

    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical instability
    return np.degrees(angle)

# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
# /               EVALUATORS                     \
# /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\
def evaluate_by_distance(behavior: pd.DataFrame) -> float:
    total_x_move = (behavior.iloc[0]['head'][1] - behavior.iloc[-1]['head'][1])

    dx = behavior['head'].apply(lambda x: x[0]) -\
            behavior['rear'].apply(lambda x: x[0])
    dy = behavior['head'].apply(lambda x: x[1]) -\
            behavior['rear'].apply(lambda x: x[1])
    
    # Compute the cosine of the angle against axis
    # We do fillna to handle division by zero if head and rear are the same
    cos_theta = dx / ((dx ** 2 + dy ** 2) **0.5).fillna(0)
    
    # Only reward alignment towards the positive axis
    avg_alignment = cos_theta.clip(lower=0).mean()
    
    return total_x_move * avg_alignment

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

        # NOTE: These angles were calculated with the vector pair method!. The
        #       angles are bounded to be between [0,180]. Therefore, it is
        #       unnecessary to account for wrapping angle values
        #       (s.t 350 and 10 = 20 degrees)
        diff1 = abs(robot_angle1 - animal_angle1)
        diff2 = abs(robot_angle2 - animal_angle2)

        return (diff1 + diff2) / 2

    behavior["Angle_Diff"] = behavior.apply(calculate_angle_difference, axis=1)

    return np.mean(behavior["Angle_Diff"])  # Return overall mean difference

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
    distances = np.array([evaluate_by_distance(behavior) 
                                for behavior in behaviors])
    alpha = data.clamp(data.inverse_lerp(gen_i, 100, 250), 0, 0.5)

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
            return [mix_ab(distance,
                           evaluate_by_angle(behavior, state.animal_data),
                            # evaluate_by_angle_dtw(behavior, state.animal_data),
                           alpha)
                    for behavior, distance in zip(behaviors, distances)]
        case _:
            raise NotImplementedError(
                f"The evaluator: `{state.similarity_type}` is not supported.")
