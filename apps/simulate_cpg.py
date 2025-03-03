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
import math
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


body = gecko_v2()


PARAMS = np.array(
    # CPG to test goes here!
    [ 2.44685766,  0.00553692,  0.70935762, -2.13680926,  0.94830019,
        0.47201074,  2.48869223,  0.04016244, -1.36223891]
       )

# This value scales the time axis of the CPG generator. 1.5 means play 50% 
# faster. Ideally, this should scale the CPG back to the animals walking speed
PLAY_SPEED = 1.5

active_hinges = body.find_modules_of_type(ActiveHinge)

(cpg_network_structure,output_mapping,) =\
    active_hinges_to_cpg_network_structure_neighbor(active_hinges)

@dataclass
class FastBrain(Brain):
    def make_instance(self) -> BrainInstance:
        return FastBrainInstance(b=BrainCpgNetworkStatic.uniform_from_params(
                    params=PARAMS,
                    cpg_network_structure=cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=output_mapping).make_instance())

@dataclass
class FastBrainInstance(BrainInstance):
    b: BrainInstance

    def control(
        self, 
        dt: float, 
        sensor_state: ModularRobotSensorState, 
        control_interface: ModularRobotControlInterface
    ) -> None:
        # Run double speed
        self.b.control(dt*PLAY_SPEED, sensor_state, control_interface)

def on_prepared() -> None:
    print("Robot is ready. Press enter to start")
    input()

robot = ModularRobot(body=body,brain=FastBrain())

print("Initializing robot..")

scene = ModularRobotScene(terrain=terrains.flat())
scene.add_robot(robot)

simulate_scenes(
    simulator=LocalSimulator(start_paused=False),
    batch_parameters=make_standard_batch_parameters(simulation_time=9999),
    scenes=scene
)
