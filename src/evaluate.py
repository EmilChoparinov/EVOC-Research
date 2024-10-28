from revolve2.modular_robot import ModularRobot

from revolve2.modular_robot_simulation import ModularRobotSimulationState

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

def evaluate(robots: list[ModularRobot], behaviors: list[simulated_behavior]) -> npt.NDArray[np.float_]:
    """
    Perform evaluation over a list of robots. The incoming data is **assumed**
    to be ordered. I.E. the first index in the modular robot list has its 
    behavior recorded in the first index of the behavior list.
    
    Returns an array of ordered fitness values.
    """
    return np.array([
        # Get delta from first state to last state in simulation
        get_pose_x_delta(
            states[0].get_modular_robot_simulation_state(robot),
            states[-1].get_modular_robot_simulation_state(robot)
        ) for robot, states in zip(robots, behaviors)
    ])

def find_most_fit(fitnesses: npt.NDArray[np.float_], robots: list[ModularRobot], behaviors: list[simulated_behavior]):
    """
    Perform linear search for the robot with the highest fitness. The incoming
    parameters are **assumed** to be ordered.
    
    Returns tuple of best fitting robot with its behavior and fitness value
    """
    # Zip fitnesses with index, find the max fitness value and index 
    fittest_idx, fitness = max(enumerate(fitnesses), key=lambda x: x[1])
    return (robots[fittest_idx],behaviors[fittest_idx], fitness)
