import numpy as np
import pandas as pd

from simulate_new import data
from simulate_new.evaluate import calculate_1_angle_single, calculate_2_angles, calculate_4_angles, \
    calculate_all_angles, calculate_1_angle_multiple

POP_SIZE = 128
animal_data_file = "./Files/animal_data_3_slow_down_lerp_2.csv"
print(f"animal_data_file: {animal_data_file}, POP_SIZE: {POP_SIZE}")

animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))[:901]
animal_individual_1_angle = np.array(calculate_1_angle_single(animal_data))
animal_population_1_angle = np.broadcast_to(animal_individual_1_angle, (POP_SIZE, 901, 1))
animal_2_angles = np.array(calculate_2_angles(animal_data))
animal_4_angles = np.array(calculate_4_angles(animal_data))
animal_all_angles = np.array(calculate_all_angles(animal_data))

def evaluate_individual_by_1_angle(robot_behavior: pd.DataFrame) -> float:
    robot_angles = np.array(calculate_1_angle_single(robot_behavior)) # .shape = (901, 1)
    simple_differences = np.abs(robot_angles - animal_individual_1_angle) # .shape = (901, 1)
    mean_differences = np.mean(simple_differences) # .shape = (1)
    score = np.mean(mean_differences) # .shape = (1)
    return score

def evaluate_population_by_1_angle(population_behaviors: list[pd.DataFrame]) -> list[float]:
    population_angles = np.array(calculate_1_angle_multiple(population_behaviors)) # .shape = (pop_size, 901, 1)
    simple_differences = np.abs(population_angles - animal_population_1_angle) # shape = (pop_size, 901, 1)
    mean_differences = np.mean(simple_differences, axis=1) # shape = (pop_size, 1)
    scores = mean_differences.flatten().tolist()
    return scores

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