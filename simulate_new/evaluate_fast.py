import numpy as np
import pandas as pd

from simulate_new import data
from simulate_new.evaluate import calculate_2_angles, calculate_4_angles, calculate_all_angles

animal_data_file = "./Files/slow_lerp_2.csv"
animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))[:901]
animal_2_angles = np.array(calculate_2_angles(animal_data))
animal_4_angles = np.array(calculate_4_angles(animal_data))
animal_all_angles = np.array(calculate_all_angles(animal_data))

def evaluate_by_2_angles(robot_behavior: pd.DataFrame) -> float:
    robot_angles = np.array(calculate_2_angles(robot_behavior)) # .shape = (901, 2)

    simple_differences = np.abs(robot_angles - animal_2_angles) # .shape = (901, 2)
    mean_differences = np.mean(simple_differences) # .shape = (2)
    score = np.mean(mean_differences) # .shape = (1)
    return score

def evaluate_by_4_angles(robot_behavior: pd.DataFrame) -> float:
    robot_angles = calculate_4_angles(robot_behavior) # .shape = (901, 4)

    simple_differences = np.abs(robot_angles - animal_4_angles) # .shape = (901, 4)
    mean_differences = np.mean(simple_differences) # .shape = (4)
    score = np.mean(mean_differences) # .shape = (1)
    return score

def evaluate_by_all_angles(robot_behavior: pd.DataFrame) -> float:
    robot_angles = np.array(calculate_all_angles(robot_behavior)) # .shape = (901, 210)

    simple_differences = np.abs(robot_angles - animal_all_angles) # .shape = (901, 210)
    mean_differences = np.mean(simple_differences) # .shape = (210)
    score = np.mean(mean_differences) # .shape = (1)
    return score