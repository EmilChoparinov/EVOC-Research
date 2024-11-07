"""
Goal:
The goal of this python file is to discover the orientation of all the joints 
in the real world and ensure they match the same orientation used in the
simulation.

The calibrator runs in two threads. One thread manages the state of the robot
while the other thread collects inputs and modifies the `inversion_map` state.
When complete, the inversion map is printed
"""

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

from revolve2.modular_robot.brain.dummy import BrainDummy

import numpy as np

import threading
from pprint import pprint

from src.config import PhysMap

robot_connection_success = threading.Condition()

body_part = Literal["left_arm", "right_arm", "left_leg", "right_leg", "tail", "torso"]

question_idx = 0
question_order: list[body_part] = ["left_arm", "right_arm", "left_leg", "right_leg", "tail", "torso"]

ornt = Literal["NORMAL", "INVERSE"]

body = gecko_v2()
body_map: dict[body_part, ActiveHingeV2] = {
        "right_arm": body.core_v2.right_face.bottom,
        "left_arm": body.core_v2.left_face.bottom,
        "torso": body.core_v2.back_face.bottom,
        "tail": body.core_v2.back_face.bottom.attachment.front,
        "right_leg":body.core_v2.back_face.bottom.attachment.front.attachment.right,
        "left_leg": body.core_v2.back_face.bottom.attachment.front.attachment.left
    }

PIN_CONFIG: dict[body_part, int] = {
    "left_arm": 0,
    "right_arm": 31,
    "left_leg": 1,
    "right_leg": 30,
    "tail": 24,
    "torso": 8
}

compact_pos = {
    0: -1,
    1: -1,
    24: 1,
    8: 1,
    30: -1,
    31: 1
}

def on_prepared() -> None:
    print("Robot is compacted!")
    exit()

brain = BrainDummy()
robot = ModularRobot(body, brain)

pmap = PhysMap.map_with(body)

config = Config(
    modular_robot=robot,
    hinge_mapping={UUIDKey(v): PIN_CONFIG[k] for k,v in body_map.items()},
    run_duration=9999,
    control_frequency=30,
    initial_hinge_positions={UUIDKey(v): compact_pos[PIN_CONFIG[k]] for k,v in body_map.items()},
    inverse_servos={v["pin"]: v["is_inverse"] for k,v in pmap.items()}
)

print("Initializing robot..")
run_remote(
    config=config,
    hostname="10.15.3.59",
    debug=False,
    on_prepared=on_prepared
)