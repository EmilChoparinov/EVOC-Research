from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotSimulationState
from animal_similarity import combination_fitnesses
from rotation_scaling import size_scaling, translation_rotation, get_data_with_forward_center
from config import alpha

import numpy as np
import numpy.typing as npt
from typedef import population, simulated_behavior, genotype

def get_pose_x_delta(state0: ModularRobotSimulationState, stateN: ModularRobotSimulationState) -> float:
    """
    Calculate the different between the starting position and the final position

    Note: Its subtracting to produce a negative value because the robot just
          happend to spawn oriented towards the -x direction
    """
    return state0.get_pose().position.x - stateN.get_pose().position.x


def evaluate(robots: list[ModularRobot], behaviors: list[simulated_behavior],df_animal,alpha,similarity_type) -> npt.NDArray[
    np.float_]:
    """
    Perform combined evaluation over a list of robots.
    Returns an array of ordered fitness values based on the combination of distance and similarity.
    """
    # Distance-based fitness
    distance_scores = np.array([
        get_pose_x_delta(
            states[0].get_modular_robot_simulation_state(robot),
            states[-1].get_modular_robot_simulation_state(robot)
        ) for robot, states in zip(robots, behaviors)
    ])
    # print("df_animal",df_animal)
    df_robot = size_scaling(translation_rotation(get_data_with_forward_center(robots, behaviors)))
    # Combination_fitnesses to get combined fitness, distance, and similarity
    combined_fitness, distance, animal_similarity= combination_fitnesses(distance_scores, df_robot,df_animal, alpha,similarity_type)

    return combined_fitness,distance,animal_similarity



# def evaluate_distance(robots: list[ModularRobot], behaviors: list[simulated_behavior]) -> npt.NDArray[np.float_]:
# #def evaluate(robots: list[ModularRobot], behaviors: list[simulated_behavior]) -> npt.NDArray[np.float_]:
#     """
#     Perform evaluation over a list of robots. The incoming data is **assumed**
#     to be ordered. I.E. the first index in the modular robot list has its
#     behavior recorded in the first index of the behavior list.
#
#     Returns an array of ordered fitness values.
#     """
#     return np.array([
#         # Get delta from first state to last state in simulation
#         get_pose_x_delta(
#             states[0].get_modular_robot_simulation_state(robot),
#             states[-1].get_modular_robot_simulation_state(robot)
#         ) for robot, states in zip(robots, behaviors)
#     ])
#
#
# def evaluate_similarity(robots: list[ModularRobot], behaviors: list[simulated_behavior]) -> npt.NDArray[np.float_]:
#     """
#     Calculate the fitness based on the similarity to a dynamic ideal behavior.
#     The ideal behavior is dynamically determined as an offset from the initial position.
#     """
#     offset_distance = 1.0  #
#     similarity_scores = []
#
#     for robot, states in zip(robots, behaviors):
#         # initial position
#         start_position = states[0].get_modular_robot_simulation_state(robot).get_pose().position
#         ideal_position = np.array([start_position.x + offset_distance, start_position.y, start_position.z])
#
#         end_position = states[-1].get_modular_robot_simulation_state(robot).get_pose().position
#         end_position_array = np.array([end_position.x, end_position.y, end_position.z])
#
#         deviation = np.linalg.norm(end_position_array - ideal_position)
#
#         similarity_score = 1 / (1 + deviation)
#         similarity_scores.append(similarity_score)
#
#     return np.array(similarity_scores)

def find_most_fit(fitnesses: npt.NDArray[np.float_], robots: list[ModularRobot], behaviors: list[simulated_behavior]):
    """
    Perform linear search for the robot with the highest fitness. The incoming
    parameters are **assumed** to be ordered.

    Returns tuple of best fitting robot with its behavior and fitness value
    """
    # Zip fitnesses with index, find the max fitness value and index 
    fittest_idx, fitness = max(enumerate(fitnesses), key=lambda x: x[1])
    return (robots[fittest_idx],behaviors[fittest_idx], fitness)
