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


import numpy as np

import threading
from pprint import pprint

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

BODY_QUESTIONS: dict[body_part, str] = {
    "left_arm": "Is left arm curved towards ground?",
    "right_arm": "Is right arm curved towards ground",
    "left_leg": "Is left leg curved towards ground?",
    "right_leg": "Is right leg curved towards ground?",
    "tail": "Is tail on body right side?",
    "torso": "Is torso on body right side?"
}

# This script generates the inversion map that matches was was scene in the 
# simulation. It maps a pin to if the signal must be inverted or not
inversion_map: dict[int, bool] = {}

@dataclass 
class CalibrationBrain(Brain):
    def make_instance(self) -> BrainInstance:
        return CalibrateBrainInstance()

@dataclass
class CalibrateBrainInstance(BrainInstance):
    def control(self, 
                dt: float,
                sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface
                ):
        # continuously send 60 degree signal to current joint in question
        question_ctx = question_order[question_idx]

        # All other hinges set to neutral
        for i, q in enumerate(question_order):
            if i == question_ctx:
                import pdb;pdb.set_trace()
                control_interface.set_active_hinge_target(
                    body_map[q], 1
                )
            #     continue
            # control_interface.set_active_hinge_target(
            #     body_map[q], 0)
             
        
def on_prepared() -> None:
    global wait_for_input
    print("Robot is ready. Press enter to start")
    input()
    
    # Wake up question interface now
    with robot_connection_success:
        robot_connection_success.notifyAll()

def question_interface() -> None:
    global question_idx
    
    # Wait for the robot to connect on other thread
    with robot_connection_success:
        robot_connection_success.wait()

    for i, q in enumerate(question_order):
        question_idx = i
        print(f"{BODY_QUESTIONS[q]} [y/n]")
        is_rev = True if input() == "y" else False
        inversion_map[PIN_CONFIG[q]] = is_rev

def connect_to_robot():
    brain = CalibrationBrain()

    robot = ModularRobot(body, brain)

    config = Config(
        modular_robot=robot,
        hinge_mapping={UUIDKey(v): PIN_CONFIG[k] for k,v in body_map.items()},
        run_duration=30,
        control_frequency=30,
        initial_hinge_positions={UUIDKey(v): 0 for _,v in body_map.items()},
        inverse_servos={},
    )

    print("Initializing robot..")
    run_remote(
        config=config,
        hostname="10.15.3.59",
        debug=True,
        on_prepared=on_prepared
    )
    
# t1 = threading.Thread(target=connect_to_robot)
# t2 = threading.Thread(target=question_interface)

# t1.start()
# t2.start()

# t1.join()
# t2.join()

connect_to_robot()
pprint(inversion_map)