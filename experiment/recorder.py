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
from revolve2.experimentation.logging import setup_logging, logging

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

from experiment.revolve_layer import remote_control_with_polling_rate
from simulate.config import PhysMap, cameras

import numpy as np

import threading
from pprint import pprint
import math
import time
from experiment.communication_layer import boot_sockets, Event, Command, Payload, Message
import experiment.config as config


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
This program is automated and generates the associated video files. If you wish
to override the automated code, focus on the camera window and press:
    q to close
    p to play/pause
    r to restart the current recording
"""
)

setup_logging()

logging.info("Activating recorder")
camera = cv2.VideoCapture(cameras[0])
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (1200, 1200)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

sio, commands, emit = boot_sockets('client')
sio.connect(f"{config.robot_ip}:{config.port}")

enable_recording = True

logging.info("Waiting for signal to record from robot")
while True:
    # Command processor:
    # Process the next command given in the queue. If the queue is empty, do
    # nothing this frame
    if len(commands) != 0:
        cmd = commands.get()
        
        match cmd.type:
            case "new_experiment":
                logging.info("Created new sample file. Must send play"
                             "command to begin recording")
                write_to_file = cv2.videowriter(
                    f"{cmd.payload.run_id}-{time.strftime('%y%m%d-%h%m%s')}.mp4", 
                    fourcc, 20.0, size)

            case "play": 
                logging.info("Now playing")
                enable_recording = True

            case "pause": 
                logging.info("Now paused")
                enable_recording = False

    # Nav controller:
    # Apply some commands via keyboard input to the buffer
    recv = cv2.waitKey(1) & 0xFF
    if recv == ord('p'):
        e = Event.pause if enable_recording else Event.play
        commands += Command(e)
        emit(Message([Command(e)]))
        
    if recv == ord('r'):
        commands += Command(Event.restart)
        commands += Command(Event.pause)
        emit(Message([Command(Event.restart)]))
    
    if recv == ord('q'):
        logging.info("Exiting")
        emit(Message([Command(Event.pause)]))
        break

    # Read/Write frame data to file
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
