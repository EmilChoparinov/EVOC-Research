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

# import config as evoc_config
from src.config import create_physical_mapping, offset_map, inversion_map



idx = 0
pins = [0,1,24,8,30,31]
body = gecko_v2()

body_map: dict[int, ActiveHingeV2] = {
        31: body.core_v2.right_face.bottom, # right arm
        0: body.core_v2.left_face.bottom, # left arm
        8: body.core_v2.back_face.bottom, # torso
        24: body.core_v2.back_face.bottom.attachment.front, # tail
        30:body.core_v2.back_face.bottom.attachment.front.attachment.right, # right leg
        1: body.core_v2.back_face.bottom.attachment.front.attachment.left # left leg
    }


@dataclass
class CalibrateHingeBrain(Brain):
    def make_instance(self):
        return CalibrateHingeBrainInstance()
    
@dataclass
class CalibrateHingeBrainInstance(BrainInstance):
    def control(self, dt, sensor_state, control_interface):
        global idx
        if(idx == len(pins)): 
            print("Calibration complete. Exiting")
            exit(0)
        print(f"Doing PIN: {pins[idx]}")
        cmd = input("Give value between [0,1] OR type `s` to skip\n")
        if(cmd.lower() == 's'): 
            idx += 1
            return
        
        
        hinge = body_map[pins[idx]]
        try:
            control_interface.set_active_hinge_target(hinge, float(cmd))
        except:
            print("Input must be a float value betweem [0,1] OR 's' to skip")
            
brain = CalibrateHingeBrain()
robot = ModularRobot(body, brain)

def on_prepared() -> None:
    print("Robot is ready. Press enter to start")
    input()
    
initial_positions = offset_map(create_physical_mapping(body))
config = Config(
    modular_robot=robot,
    hinge_mapping={UUIDKey(v): k for k,v in body_map.items()},
    run_duration=99999,
    control_frequency=30,
    # initial_hinge_positions=initial_positions,
    initial_hinge_positions={UUIDKey(v): 0 for k,v in body_map.items()},
    inverse_servos=inversion_map(create_physical_mapping(body)),
)

print("Initializing robot..")
run_remote(
    config=config,
    hostname="10.15.3.59",
    debug=True,
    on_prepared=on_prepared
)
