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
from flask import Flask
import time
from revolve2.experimentation.logging import setup_logging, logging

#=== Config
camera_ip   = ""
output_size = (1200, 1200)                    # Specify file size
fourcc      = cv2.VideoWriter_fourcc(*'mp4v') # Specify codec

#=== Data Management
# This file manages 3 threads over the enclosed object below. 
# - From network
# - From keyboard
# - To the camera

@dataclass
class CameraState():
    """
    exit_signal:
        If true, notifies all threads to exit
    is_recording:
        If true, writes out the camera feed to current file
    open_signal:
        If true, close current file, and write to new file
    """
    exit_signal: bool
    is_recording: bool
    open_signal: bool
    
    filename: str

#=== Setup Routine
start_time = datetime.now()
timestr = f"{now:%d/%m/%y-%H:%M}"
setup_logging(f"recorder-log-{timestr}.txt")
logging.info(
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
This is an **automated** recorder that will handle recording the gecko robot in
the designated zone. This script acts as a **server** where the robot queries
when to stop a recording and begin a new one. You can override the robots 
actions by pressing the keys described.

Press while focused on camera window to override default behaviors:
    q to close
    n to start next recording immediately
    s to stop the current recording. Press again to continue!

"""
)

logging.info("Connecting to recorder...")
camera = cv2.VideoCapture(camera_ip)

# Crop the recording because the camera is warped with a fisheye lense. So we
# zoom into the middle of the recording.
width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

buf_write_out = cv2.\
    VideoWriter(f"{time.strftime('%Y%m%d-%H%M%S')}.mp4", fourcc, 20.0, size)

logging.info("Starting web server...")

app = Flask(__name__)

@app.route("/cmd/next")
def recv_signal_next():
    pass

@app.route("/cmd/pause")
def recv_signal_pause():
    pass

@app.route("/cmd/play")
def recv_signal_play():
    pass

@app.route("/set_label")
def recv_signal_label():
    pass

# print("=== RECORDER IS ACTIVE ===")
# camera = cv2.VideoCapture(cameras[0])
# size = (1200, 1200)

# write_to_file = cv2.VideoWriter(f"{time.strftime('%Y%m%d-%H%M%S')}.mp4", fourcc, 20.0, size)

# enable_recording = True
# while True:
#     # Nav controller
#     recv = cv2.waitKey(1) & 0xFF
#     if recv == ord('n'):
#         print("=== Recording finished. Starting new recording!")
#         write_to_file = cv2.VideoWriter(
#             f"{time.strftime('%Y%m%d-%H%M%S')}.mp4", fourcc, 20.0, size)
#         enable_recording = True
#     elif recv == ord('q'):
#         print("=== Processing closing request. Please stand by")
#         break
#     elif recv == ord('s'):
#         enable_recording = not enable_recording
#         print("Now playing" if enable_recording else "Stopped")

#     # Read frame data
#     ret, frame = camera.read()
#     if not ret:
#         print("Fatal Error: Failed to get next camera frame")
#         exit(1)
    
#     frame = frame[:, 300:1620]
#     frame = cv2.resize(frame, size)

#     cv2.imshow("Camera Feed", frame)
#     if enable_recording: write_to_file.write(frame)

# camera.release()
# cv2.destroyAllWindows()
# print("Success!")