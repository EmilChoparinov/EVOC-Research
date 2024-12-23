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
from revolve2.modular_robot.brain.dummy import BrainDummy

import pandas as pd
import numpy as np

rng = make_rng_time_seed()

robot_body = gecko_v2()


robot_brain = BrainDummy()
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