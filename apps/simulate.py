"""
Goal:
The goal of this python file is to load the genotype into the simulation to 
observe the robot moving manually.
"""

import argparse

from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.standards import terrains
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor, BrainCpgNetworkNeighborRandom, CpgNetworkStructure, BrainCpgNetworkStatic
from revolve2.standards.modular_robots_v2 import gecko_v2, snake_v2
from revolve2.modular_robot.body.base import ActiveHinge

import pandas as pd
import numpy as np

# Set up the argument parser
parser = argparse.ArgumentParser(description="Script that simulates parameters in the simulation for viewing")

# The default or set your own!
weight_matrix_csv_filepath = "./EA/best-cpg-gen-4.csv" 
parser.add_argument("weights", default=weight_matrix_csv_filepath, type=str, help="Path to weight matrix")


# Parse the arguments
args = parser.parse_args()

rng = make_rng_time_seed()

robot_body = gecko_v2()

# TODO: Bug here. Only the hack works for now
# # Technically this should be ActiveHingeV2 but this is the super class.
# active_hinges = robot_body.find_modules_of_type(ActiveHinge)
# (
#     cpg_net_struct, output_mapping
# ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

# # We construct a static brain such that the csv is used as the weight matrix
# robot_brain = BrainCpgNetworkStatic(
    
#     # Set CPG network to have a uniform neutral state (all set to 0)
#     initial_state=cpg_net_struct.make_uniform_state(0),
    
#     # Perform a transformation from csv :-> pdarray :-> ndarray
#     weight_matrix=pd.read_csv(weight_matrix_csv_filepath, header=None).to_numpy(dtype=float),
#     output_mapping=output_mapping
# )

# We overwrite the random CPG with our own weight matrix
robot_brain = BrainCpgNetworkNeighborRandom(robot_body, rng) 
# Perform matrix transform from the csv. Flip the horizontal. Then convert from pdarray :-> ndarray.
robot_brain._weight_matrix = pd.read_csv(args.weights, header=None).to_numpy(dtype=float)

# from revolve2.modular_robot.brain.dummy import BrainDummy 

# robot_brain = BrainDummy()
robot = ModularRobot(robot_body, robot_brain)

def main() -> None:
    setup_logging()
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)
    
    simulate_scenes(
        simulator=LocalSimulator(start_paused=False),
        batch_parameters=make_standard_batch_parameters(simulation_time=999999),
        scenes=scene
    )

if __name__ == "__main__": main()