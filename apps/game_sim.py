"""
Goal:
Use masking techniques on CPGs to make robot uni-directional!
"""

import pygame

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
from revolve2.modular_robot.brain.cpg import BrainCpgInstance, active_hinges_to_cpg_network_structure_neighbor, BrainCpgNetworkNeighborRandom, CpgNetworkStructure, BrainCpgNetworkStatic
from revolve2.standards.modular_robots_v2 import gecko_v2, snake_v2
from revolve2.modular_robot.body.base import ActiveHinge

import pandas as pd
import numpy as np
import numpy.typing as npt

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
import math

# Initialize pygame to interface between a real world controller
import pygame
import time

pygame.init()

while True:
    # Process Pygame events
    for event in pygame.event.get():
        # Check for keydown events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                print("W is pressed")
            elif event.key == pygame.K_a:
                print("A is pressed")
            elif event.key == pygame.K_s:
                print("S is pressed")
            elif event.key == pygame.K_d:
                print("D is pressed")
            elif event.key == pygame.K_LEFT:
                print("Left Arrow is pressed")
            elif event.key == pygame.K_RIGHT:
                print("Right Arrow is pressed")
            elif event.key == pygame.K_q:
                print("Exiting...")
                pygame.quit()
                exit()

    time.sleep(0.01)  # Avoid busy-waiting

exit()

params = {
    "left_rot": np.array(
[-2.41680794, -0.09332698, -2.01586048, -0.22636505, -1.9637401 ,
        2.4865368 ,  0.32540607, -1.05750314,  2.49999121]
            ), 
    "right_rot": np.array(
[-2.13869271,  0.00726176, -1.77607038,  1.72366607, -2.44596507,
       -2.30199428, -1.103323  ,  0.60124181,  0.44488595]
            ),
    "left": np.array(
                [-2.41618065, -0.09473266, -2.02591035, 
                -0.22562266, -1.96329705, 2.48678446,  
                0.32659213, -1.05492081,  2.49996295]
            ),
    "back": np.array(
                [-2.41618065, -0.09473266, -2.02591035, 
                -0.22562266, -1.96329705, 2.48678446,  
                0.32659213, -1.05492081,  2.49996295]
            ),
    "forward": np.array(
                [-2.41618065, -0.09473266, -2.02591035, 
                -0.22562266, -1.96329705, 2.48678446,  
                0.32659213, -1.05492081,  2.49996295]
            ),
    "right": np.array(
                [-2.41618065, -0.09473266, -2.02591035, 
                -0.22562266, -1.96329705, 2.48678446,  
                0.32659213, -1.05492081,  2.49996295]
            ),
}

body = gecko_v2()

active_hinges = body.find_modules_of_type(ActiveHingeV2)

(
    cpg_net_struct,
    output_map
) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

brain_map = {
    label: BrainCpgNetworkStatic.uniform_from_params(
                params=cpg,
                cpg_network_structure=cpg_net_struct,
                initial_state_uniform=math.sqrt(2) * 0.5,
                output_mapping=output_map
            ).make_instance()
    for label,cpg in params.items()
}

@dataclass
class BrainController(Brain):
    def make_instance(self) -> BrainInstance:
        return BrainControllerInstance(
            state=math.sqrt(2) * 0.5
        )
acc = 0
class BrainControllerInstance(BrainInstance):
    def __init__(
        self, state: npt.NDArray[np.float_]
    ):
        self._state = cpg_net_struct.make_uniform_state(state)
    
    def control(
        self, 
        dt: float, 
        sensor_state: ModularRobotSensorState, 
        control_interface: ModularRobotControlInterface
    ) -> None:
        global acc
        # The trick is to override the internal CPG state with the overall
        # state being tracked in this function. The idea here is that given
        # any previous state, the CPG might be able to naturally figure out
        # what to do

        # acc += dt
        # if(acc > 2):
        #     right_stick_x = -1
        # else: right_stick_x = 1
        
        # if acc > 4: acc = 0

        # left_stick_x = joystick.get_axis(0)
        # left_stick_y = joystick.get_axis(1)
        # print(f"Left Stick - X: {left_stick_x:.2f}, Y: {left_stick_y:.2f}")

        # # We only care about the x axis on the stick
        # right_stick_x = joystick.get_axis(2)
        # print(f"Right Stick - X: {right_stick_x:.2f}")

        # select_brain = brain_map["left_rot"]
        # if(right_stick_x > 0):
        #     select_brain = brain_map["right_rot"]

        select_brain = brain_map["left_rot"]
        if(keyboard.is_pressed('right_rot')):
            select_brain = brain_map["right_rot"]

        select_brain.control(dt, sensor_state, control_interface)

        # self._state = select_brain._rk45(self._state, select_brain._weight_matrix, dt)
        
        # for state_index, active_hinge in select_brain._output_mapping:
        #     control_interface.set_active_hinge_target(
        #         active_hinge, float(self._state[state_index]) * active_hinge.range
        #     )

robot = ModularRobot(
    body=body,
    brain=BrainController(),
)

pmap = PhysMap.map_with(body)
config = Config(
    modular_robot=robot,
    hinge_mapping={UUIDKey(v["hinge"]): v["pin"] for k,v in pmap.items()},
    run_duration=99999,
    control_frequency=30,
    initial_hinge_positions={UUIDKey(v["hinge"]): 0 for k,v in pmap.items()},
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