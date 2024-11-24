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

from dataclasses import dataclass
from typing import Literal

from pyrr import Vector3

from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2
from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.body.sensors import CameraSensor
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote

from src.network_layer import remote_control_with_polling_rate
from src.config import PhysMap

import numpy as np

import threading
from pprint import pprint

body = gecko_v2()
body_map: dict[int, ActiveHingeV2] = {
        31: body.core_v2.right_face.bottom, # right arm
        0: body.core_v2.left_face.bottom, # left arm
        8: body.core_v2.back_face.bottom, # torso
        24: body.core_v2.back_face.bottom.attachment.front, # tail
        30:body.core_v2.back_face.bottom.attachment.front.attachment.right, # right leg
        1: body.core_v2.back_face.bottom.attachment.front.attachment.left # left leg
    }

# Set up the argument parser
# parser = argparse.ArgumentParser(description="Script that simulates parameters in the simulation for viewing")

# # The default or set your own!
# weight_matrix_csv_filepath = "./EA/best-cpg-gen-4.csv" 
# parser.add_argument("weights", default=weight_matrix_csv_filepath, type=str, help="Path to weight matrix")

# Parse the arguments
# args = parser.parse_args()

rng = make_rng_time_seed()

# We overwrite the random CPG with our own weight matrix
robot_brain = BrainCpgNetworkNeighborRandom(body, rng) 
# Perform matrix transform from the csv. Flip the horizontal. Then convert from pdarray :-> ndarray.
# robot_brain._weight_matrix = pd.read_csv(args.weights, header=None).to_numpy(dtype=float)

PARAMS = np.array(
   [-2.99999447,  1.81308186, -0.0331997 ,  0.06852362,  2.89306811,
       -0.58302268,  2.76108739, -2.99385889, -1.80114772]
)



# from revolve2.modular_robot.brain.dummy import BrainDummy 

# robot_brain = BrainDummy()
# robot = ModularRobot(body, robot_brain)

active_hinges = body.find_modules_of_type(ActiveHinge)

(
    cpg_network_structure,
    output_mapping,
) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)


import math

robot = ModularRobot(
                body=body,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=PARAMS,
                    cpg_network_structure=cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=output_mapping,
                ),
            )

def on_prepared() -> None:
    print("Robot is ready. Press enter to start")
    input()
pmap = PhysMap.map_with(body)
config = Config(
    modular_robot=robot,
    hinge_mapping={UUIDKey(v): k for k,v in body_map.items()},
    run_duration=99999,
    control_frequency=30,
    initial_hinge_positions={UUIDKey(v): 0 for k,v in body_map.items()},
    inverse_servos={v["pin"]: v["is_inverse"] for k,v in pmap.items()},
)

print("Initializing robot..")
remote_control_with_polling_rate(
    config=config,
    port=20812,
    hostname="10.15.3.59",
    rate=10
)

# run_remote(
#     config=config,
#     hostname="10.15.3.59",
#     debug=True,
#     on_prepared=on_prepared
# )
