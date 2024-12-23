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
from typing import Callable, Literal, TypedDict

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


rng = make_rng_time_seed()

robot_brain = BrainCpgNetworkNeighborRandom(body, rng) 

PARAMS = np.array(
    [1.99079952, 0.40551454, 0.13197394, -1.13272489, -1.9787915, -1.20571256, 1.54274854, -0.01079462, -0.61889371]
#    [-2.99999447,  1.81308186, -0.0331997 ,  0.06852362,  2.89306811,
#        -0.58302268,  2.76108739, -2.99385889, -1.80114772]
    # [-1.99028127,  0.08435464,  0.00391817, -0.02174904,  1.94509952,
    #     1.60096773,  1.61155222, -1.99356261, -1.12994603]
)


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

phys_box_names = Literal["left_arm", "right_arm", "torso", "tail", "left_leg", "right_leg"]
class PhysMap(TypedDict):
    pin: int
    extract: Callable[[BodyV2], ActiveHingeV2]

    def get_box(body: BodyV2,box: phys_box_names):
        phy_map: dict[phys_box_names, ActiveHingeV2] = {
            "left_arm": body.core_v2.left_face.bottom.attachment,
            "left_leg": body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment,
            "torso": body.core_v2.back_face.bottom.attachment,
            "right_arm": body.core_v2.right_face.bottom.attachment,
            "right_leg":body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment,
            "tail": body.core_v2.back_face.bottom.attachment.front.attachment,
        }
        return phy_map[box]
    
    def map_with(body: BodyV2) -> dict[phys_box_names, 'PhysMap']:
        return {
            "left_arm": {
                "pin": 0,
                "hinge": PhysMap.get_box(body, "left_arm"),
                "is_inverse": True
            },
            "left_leg": {
                "pin": 1,
                "hinge": PhysMap.get_box(body, "right_arm"),
                "is_inverse": False
            },
            "torso": {
                "pin": 8,
                "hinge": PhysMap.get_box(body, "torso"),
                "is_inverse": False
            },
            "right_arm": {
                "pin": 31,
                "hinge": PhysMap.get_box(body, "right_arm"),
                "is_inverse": False
            },
            "right_leg": {
                "pin": 30,
                "hinge": PhysMap.get_box(body, "right_leg"),
                "is_inverse": True
            },
            "tail": {
                "pin": 24,
                "hinge": PhysMap.get_box(body, "tail"),
                "is_inverse": False
            },
        }
 

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

scene = ModularRobotScene(terrain=terrains.flat())
scene.add_robot(robot)

simulate_scenes(
    simulator=LocalSimulator(start_paused=False),
    batch_parameters=make_standard_batch_parameters(simulation_time=9999),
    scenes=scene
)
