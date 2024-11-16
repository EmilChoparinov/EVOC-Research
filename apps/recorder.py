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
import threading
import numpy as np
import cv2

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
from src.config import PhysMap, cameras

import numpy as np

import threading
from pprint import pprint
import math
import time

print(
"""

                                   _..._                 .-'''-.           _..._       .-'''-.                                                ___   
                                .-'_..._''.             '   _    \      .-'_..._''.   '   _    \ _______                                   .'/   \  
              __.....__       .' .'      '.\    .     /   /` '.   \   .' .'      '.\/   /` '.   \\  ___ `'.         __.....__             / /     \ 
  .--./)  .-''         '.    / .'             .'|    .   |     \  '  / .'          .   |     \  ' ' |--.\  \    .-''         '.           | |     | 
 /.''\\  /     .-''"'-.  `. . '             .'  |    |   '      |  '. '            |   '      |  '| |    \  '  /     .-''"'-.  `. .-,.--. | |     | 
| |  | |/     /________\   \| |            <    |    \    \     / / | |            \    \     / / | |     |  '/     /________\   \|  .-. ||/`.   .' 
 \`-' / |                  || |             |   | ____`.   ` ..' /  | |             `.   ` ..' /  | |     |  ||                  || |  | | `.|   |  
 /("'`  \    .-------------'. '             |   | \ .'   '-...-'`   . '                '-...-'`   | |     ' .'\    .-------------'| |  | |  ||___|  
 \ '---. \    '-.____...---. \ '.          .|   |/  .                \ '.          .              | |___.' /'  \    '-.____...---.| |  '-   |/___/  
  /'""'.\ `.             .'   '. `._____.-'/|    /\  \                '. `._____.-'/             /_______.'/    `.             .' | |       .'.--.  
 ||     ||  `''-...... -'       `-.______ / |   |  \  \                 `-.______ /              \_______|/       `''-...... -'   | |      | |    | 
 \'. __//                                `  '    \  \  \                         `                                                |_|      \_\    / 
  `'---'                                   '------'  '---'                                                                                  `''--'  

How to use!
Press while focused on camera window:
    q to close
    n to start next recording immediately
    s to stop the current recording. Press again to continue!
"""
)

print("=== RECORDER IS ACTIVE ===")
camera = cv2.VideoCapture(cameras[0])
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (1200, 1200)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

write_to_file = cv2.VideoWriter(f"{time.strftime('%Y%m%d-%H%M%S')}.mp4", fourcc, 20.0, size)

enable_recording = True
while True:
    # Nav controller
    recv = cv2.waitKey(1) & 0xFF
    if recv == ord('n'):
        print("=== Recording finished. Starting new recording!")
        write_to_file = cv2.VideoWriter(
            f"{time.strftime('%Y%m%d-%H%M%S')}.mp4", fourcc, 20.0, size)
        enable_recording = True
    elif recv == ord('q'):
        print("=== Processing closing request. Please stand by")
        break
    elif recv == ord('s'):
        enable_recording = not enable_recording
        print("Now playing" if enable_recording else "Stopped")

    # Read frame data
    ret, frame = camera.read()
    if not ret:
        print("Fatal Error: Failed to get next camera frame")
        exit(1)
    
    frame = frame[:, 300:1620]
    frame = cv2.resize(frame, size)

    cv2.imshow("Camera Feed", frame)
    if enable_recording: write_to_file.write(frame)

camera.release()
cv2.destroyAllWindows()
print("Success!")