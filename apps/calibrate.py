"""
Goal:
The goal of this python file is to discover the orientation of all the joints 
in the real world and ensure they match the same orientation used in the
simulation
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

from ..src.config import body_to_csv_map

body_part = Literal["left_arm", "right_arm", "left_leg", "right_leg", "tail", "torso"]
ornt = Literal["NORMAL", "INVERSE"]

body = gecko_v2()

PIN_CONFIG: dict[body_part, int] = {
    "left_arm": 0,
    "right_arm": 31,
    "left_leg": 1,
    "right_leg": 30,
    "tail": 24,
    "torso": 8
}

BODY_QUESTIONS: dict[body_part, str] = {
    "left_arm": "Is left arm curved towards ground?",
    "right_arm": "Is right arm curved towards ground",
    "left_leg": "Is left leg curved towards ground?",
    "right_leg": "Is right leg curved towards ground?",
    "tail": "Is tail on body right side?",
    "torso": "Is torso on body right side?"
}

@dataclass 
class CalibrationBrain(Brain):
    pass

