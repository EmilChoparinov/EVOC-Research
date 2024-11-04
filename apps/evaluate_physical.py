"""
Goal:
The goal of this python file is to run various CPGs through a single brain in
one iteration. You should perform whatever measurements and recordings you want
of these CPGs
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
import math

# All of these are recorded at 200 generations
batch_cpg_params = [
    [-2.47124768,  0.17264671,  0.30393733, -0.12812158, -1.27522738,
       -0.32205638,  2.25565595, -2.4977165 , -1.62010596], # -1.61
    [-2.49991299,  0.44331288, -0.01261823, -0.01098447,  0.13277421,
       -2.4828977 ,  2.30644179, -2.1834448 , -1.42267434], # ?
    [-1.98907633, -0.21464625,  0.00993038, -0.0145763 ,  1.46751641,
            1.65157093,  1.98277571, -1.93597174, -1.1197288 ], # -1.91
    [-1.98942131,  1.27752048, -0.01371077, -1.99251466, -0.34594037,
       -1.3240709 ,  1.56589035, -1.57862379, -1.0269449 ], # -1.38
    [-1.99671427e+00,  4.54036835e-01, -9.78451406e-03,  4.62432356e-04,
        1.06113886e+00,  1.86055620e+00,  1.59448163e+00, -1.99802220e+00,
       -1.42364389e+00] # 1.85
]

body = gecko_v2()

active_hinges = body.find_modules_of_type(ActiveHingeV2)
(
    cpg_net_struct,
    output_map
) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

brains = [
    BrainCpgNetworkStatic.uniform_from_params(
                params=cpg,
                cpg_network_structure=cpg_net_struct,
                initial_state_uniform=math.sqrt(2) * 0.5,
                output_mapping=output_map
    ).make_instance()
    for cpg in batch_cpg_params
]

@dataclass
class BatchTesterBrain(Brain):
    def make_instance(self) -> BrainInstance:
        return BatchTesterBrainInstance(
            hinges=active_hinges,
            brains=brains
        )

@dataclass
class BatchTesterBrainInstance(BrainInstance):
    hinges: list[ActiveHinge]
    brains: list[BrainInstance]
    # Time since 0 (when run starts)
    dt0: float = 0
    idx: int = 0
    
    # We pause the robots movements between experiments
    capture_dt: bool = True
    
    def control(
        self, 
        dt: float, 
        sensor_state: ModularRobotSensorState, 
        control_interface: ModularRobotControlInterface
    ) -> None:
        if self.capture_dt:
            self.dt0 += dt
        else:
            self.capture_dt = True
        
        # After 30 seconds, we progress to the next CPG
        if(self.dt0 > 5):
            self.idx += 1
            # If idx reached the end, quit
            import pdb;pdb.set_trace()
            if(self.idx == len(self.brains)):
                print("Test complete. Shutting down")
                exit()
            print("Loading next...")
            [control_interface.set_active_hinge_target(h, 0) for h in self.hinges]
            input(f"Loaded CPG Index: {self.idx}. Press enter to start test next test")
            
            # Reset dt states
            self.dt0 = 0
            self.capture_dt = True
            return

        self.brains[self.idx].control(dt, sensor_state, control_interface)

robot = ModularRobot(body=body,
                     brain=BatchTesterBrain()
)

def on_prepared() -> None:
    print("Robot is ready. Press enter to start")
    input()
pmap = PhysMap.map_with(body)
body_map: dict[int, ActiveHingeV2] = {
        31: body.core_v2.right_face.bottom, # right arm
        0: body.core_v2.left_face.bottom, # left arm
        8: body.core_v2.back_face.bottom, # torso
        24: body.core_v2.back_face.bottom.attachment.front, # tail
        30:body.core_v2.back_face.bottom.attachment.front.attachment.right, # right leg
        1: body.core_v2.back_face.bottom.attachment.front.attachment.left # left leg
    }
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