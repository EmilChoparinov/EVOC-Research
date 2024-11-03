"""
Goal:
Use masking techniques on CPGs to make robot uni-directional!
"""

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

from dataclasses import dataclass, field
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

from collections import namedtuple, deque

import numpy as np
import math
from pprint import pprint
from functools import reduce
import numpy as np

# Initialize pygame to interface between a real world controller
import pygame.joystick as joystick
import pygame
import time
import os

pygame.display.set_mode(size=(320,320))
joystick.init()

if(pygame.joystick.get_count() == 0):
    raise EnvironmentError("IO FAILURE: Cannot find controller input. Maybe check cable or connection?")

def select_joy_routine() -> joystick.Joystick:
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    
    # Print each joystick with their ID:
    [print(f"{joy.get_id()}: {joy.get_name()}") for joy in joysticks]
    if(len(joysticks) == 1): 
        print(f"Connected with {joysticks[0]}. Was only option")
        return joysticks[0]

    id = input("Select Controll ID from above: ")
    if not id.isdigit() and int(id) < 0 or int(id) >= len(joysticks):
        print("Invalid ID, try again")
        return select_joy_routine()
    return joysticks[int(id)]

JoyState = namedtuple('JoyState', ['left_stick_x', 'left_stick_y', 'right_stick_x', 'right_stick_y'])
def get_joy_state(joy: joystick.Joystick) -> JoyState:
    pygame.event.get()
    return JoyState(
        joy.get_axis(0),
        joy.get_axis(1),
        joy.get_axis(3),
        joy.get_axis(4)
    )

joy = select_joy_routine()

params = {
    "left_rot": np.array(
[ 0.13795819, -2.49202708, -0.56547576,  0.57398807, -0.68519928,
       -1.58036373, -2.4154401 , -1.24528933, -1.577584  ]
# [ 0.75052009, -0.39783284,  1.09146232,  0.5567706 ,  1.0520274 ,
#        -0.49341066,  0.62598813,  0.81604875,  0.38351799]
            ), 
    "right_rot": np.array(
[-2.13869271,  0.00726176, -1.77607038,  1.72366607, -2.44596507,
       -2.30199428, -1.103323  ,  0.60124181,  0.44488595]
# [-0.64907861, -0.20690301, -0.96089514,  0.92607873, -1.27081019,
#         0.70718109,  1.16130489, -1.22530135,  1.54471567]
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
   [-2.99999447,  1.81308186, -0.0331997 ,  0.06852362,  2.89306811,
       -0.58302268,  2.76108739, -2.99385889, -1.80114772]
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
            state=cpg_net_struct.make_uniform_state(math.sqrt(2) * 0.5),
            cpg_state=brain_map["left_rot"]
        )

@dataclass
class BrainControllerInstance(BrainInstance):

    state: npt.NDArray[np.float_]
    cpg_state: BrainInstance
    # The transition buffer contains a list of states that the hinges will
    # transfer to at polling rate
    transition_buffer: deque[list[tuple[int, ActiveHinge]]] = field(default_factory=deque) 

    def control(
        self, 
        dt: float, 
        sensor_state: ModularRobotSensorState, 
        control_interface: ModularRobotControlInterface
    ) -> None:
        pygame.event.get()

        joy_state = get_joy_state(joy)
        
        select_brain = self.cpg_state

        if(joy_state.right_stick_x < 0.1 and joy_state.right_stick_x > -0.1):
            select_brain = brain_map["forward"]
            print("go forward")
        elif(joy_state.right_stick_x > 0):
            select_brain = brain_map["right_rot"]
            print("go right")
        elif(joy_state.right_stick_x < 0):
            select_brain = brain_map["left_rot"]
            print("go left")

        # If there exists an item in the buffer, pop an element and use that
        # As the next hinge configuration for this control call and return
        if len(self.transition_buffer) > 0:
            next_state = self.transition_buffer.popleft()
            [control_interface.set_active_hinge_target(hinge, target) 
            for target, hinge in next_state]
            return

        # If the state has transitioned, new positions into buffer to slowly 
        # sync/interpolate between now and after.
        # NOTE: All brains use the SAME output mapping. This means we have 
        #       guarenteed an order
        if select_brain != self.cpg_state:
            select_brain._state = select_brain._rk45(select_brain._state, select_brain._weight_matrix, dt)
            current_positions = [
                (float(self.cpg_state._state[state_idx]) * hinge.range, hinge) 
                for state_idx, hinge in  self.cpg_state._output_mapping
            ]
        
            next_positions = [
                        (float(select_brain._state[state_idx]) * hinge.range, hinge) 
                        for state_idx, hinge in  select_brain._output_mapping
                    ]

            # Assert that the hinge are ordered in the same way between both 
            # lists
            assert(all(next_positions[i][1] == current_positions[i][1] 
                        for i in range(len(next_positions))))

            # np.linspace()
            linpol_positions_hinge = [
                (np.linspace(c_hinge_pos, n_hinge_pos,5), c_hinge) 
                for (c_hinge_pos, c_hinge), (n_hinge_pos, n_hinge) in 
                zip(current_positions, next_positions)
            ]

            hinge_positions_linpol = [pos for pos,_ in linpol_positions_hinge]
            position_states = [state for state in zip(*hinge_positions_linpol)]
            pos_states_and_hinge = [
                list(
                    map(lambda i: (i[1], select_brain._output_mapping[i[0]][1]), 
                    enumerate(state))) 
                for state in position_states]

            self.transition_buffer.extend(pos_states_and_hinge)
            
            # Perform state transition
            self.cpg_state = select_brain
            return

        # If the buffer is empty and there was no state transition, we can just 
        # transfer the state to the next one in the cpg like normal             
        select_brain.control(dt, sensor_state, control_interface)

robot = ModularRobot(
    body=body,
    brain=BrainController(),
)

pmap = PhysMap.map_with(body)
config = Config(
    modular_robot=robot,
    hinge_mapping={UUIDKey(v["box"]): v["pin"] for k,v in pmap.items()},
    run_duration=99999,
    control_frequency=30,
    initial_hinge_positions={UUIDKey(v["box"]): 0 for k,v in pmap.items()},
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