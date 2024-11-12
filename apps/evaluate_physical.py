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

# CondLocks used to communicate between the remote control robot thread and the
# filming thread
on_run_finished = threading.Event()
on_run_start = threading.Condition()

# These CPG params are ordered from the 30-runs-300-gen-distance-only.zip file
batch_cpg_params = [
    [-1.99996053, -0.25339486,  0.86433392,  0.04125646,  1.99781591,
            1.79550083,  1.99910857, -1.77946732, -1.3014065 ],
    [ 1.28400412,  1.01023474,  0.24121196,  0.17842347, -0.56293134,
    0.24181981,  1.9997911 , -0.03222038,  0.68095533],
    [-1.98058858, -0.01903976, -1.6223547 , -1.73164397, -0.56285298,
       -0.14980731,  1.8613266 , -1.99676837, -0.93451145],
    [ 1.99467832e+00,  1.42880505e+00, -4.94073073e-01,  1.53958940e+00,
        1.54854010e+00, -1.54087307e+00,  1.92305615e+00,  4.77303346e-04,
       -5.23782691e-01],
    [-0.37245841,  0.50721665,  1.8061204 , -1.31451915, -1.46352754,
       -1.95947652,  1.621419  , -1.93092012, -0.86679466],
    [-1.99865145,  0.94180217,  0.62508605, -0.67133064, -1.43048869,
        -1.56804528,  1.74588369, -1.95970235, -1.65708418],
    [-1.99737207e+00,  7.43812707e-01, -1.19135152e-02,  1.23735071e-03,
        1.99071857e+00,  1.06532455e+00,  1.59893729e+00, -1.60567527e+00,
       -1.11415614e+00],
    [-1.9998925 ,  0.12992104,  0.002463  , -0.01660676,  0.73585466,
        1.62312859,  1.6496734 , -1.99764393, -1.23819509],
    [ 1.46273813, -1.26619314,  0.11744505, -1.45489023, -0.72264335,
    -0.75016866, -0.10613329, -0.02808386, -1.90498608],
    [-1.99869601,  1.00810971, -0.02847164, -1.99999624, -0.09664339,
       -1.22156122,  1.99882625, -1.9047915 , -1.10945452],
    [ 1.98911371,  0.0515929 , -0.61143619,  0.26842116, -1.18795901,
       -0.11794659,  1.7495822 ,  0.21807973, -1.56900356],
    [-1.99757566,  0.14862276, -0.01154607, -0.01276641, -0.3776174 ,
       -1.98704461,  1.62551093, -1.89799684, -1.26867704],
    [ 1.061951  ,  1.95597436,  0.97490947, -1.9851532 ,  1.71373849,
        1.00907185,  1.82187735, -0.32475766, -1.44546651],
    [-1.82230499,  1.35508709, -1.99990645, -1.93247915, -0.05296935,
        1.23557556,  1.48603491, -1.95707487, -0.81206936],
    [ 1.70999838,  1.19567045,  0.65525818,  0.03602676, -1.71326936,
       -0.90489436,  1.44693539, -0.01387267, -1.12031079],
    [-1.99779011,  0.87326926,  0.61641282, -0.75244288, -1.34263858,
       -1.96207764,  1.99902096, -1.99971423, -1.42870028],
    [-1.97805837,  0.10714043, -0.0125334 ,  0.00290491, -1.0307521 ,
       -1.93821673,  1.69584984, -1.79262588, -1.21422434],
    [-1.96026348,  1.92396499,  1.38933883, -0.82054481,  0.73658213,
       -1.6418812 ,  1.7605764 , -1.97125958, -1.97235403],
    [-1.99384599,  1.28172728, -0.02433415, -1.99998186, -1.85127478,
       -1.99996542,  1.70509135, -1.75849619, -0.97699127],
    [ 1.84338986, -0.44095548,  0.36124198, -1.9826054 ,  1.96149659,
        0.64715368,  1.85251222, -1.78443963, -1.70971335],
    [-1.9664299 ,  0.25831549,  0.91003017, -0.68513369,  1.55217076,
       -1.67080068,  1.94436093, -1.40580438, -1.72625616],
    [-1.99254696,  0.68350221,  0.02298475, -1.99798187,  0.36487141,
       -1.99785532,  1.999162  , -1.70178275, -1.07724027],
    [-1.99927537,  0.64369408,  0.09209359, -1.99744344, -0.0544267 ,
        1.99551393,  1.75669445, -1.75821245, -1.15073075],
    [-1.99249018,  0.03529181, -0.00594795, -0.01618056, -0.5523184 ,
       -1.43426456,  1.52657583, -1.81621956, -1.09618642],
    [-1.9443146 , -0.00970808, -1.594218  , -1.86408067,  0.79658922,
        1.13362529,  1.69960363, -1.98134217, -1.2711723 ],
    [-0.43704218,  0.55256768, -1.99891374, -1.62256924,  1.79835549,
        1.98654758,  1.98095494, -1.99432017, -0.53927368],
    [-1.97604487e+00,  2.43544133e-01, -1.17776776e-03, -2.94365539e-02,
        1.93145384e+00, -1.13669486e-01,  1.84145878e+00, -1.99994414e+00,
       -1.14948977e+00],
    [-1.99982617,  0.13715334,  0.0273956 , -0.0337289 , -1.99640923,
        1.99997355,  1.65602231, -1.5899361 , -1.29715043],
    [-1.99999995,  0.15997493,  0.01081991, -0.03458802, -1.70696704,
       -1.23867053,  1.99541639, -1.99861173, -1.3124398 ],
    [-1.6459108 ,  0.57498081,  0.00844196, -1.99963482,  0.81886104,
       -1.7708486 ,  0.89132614, -1.50606673, -0.86288779]
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
    
    # Yup, blame netcode not me
    spam_times: int = 20

    def control(
        self, 
        dt: float, 
        sensor_state: ModularRobotSensorState, 
        control_interface: ModularRobotControlInterface
    ) -> None:
        if self.spam_times > 0:
            [control_interface.set_active_hinge_target(h, 0) for h in self.hinges]
            self.spam_times -= 1
            return
        
        if self.capture_dt:
            self.dt0 += dt
        else:
            self.capture_dt = True
            with on_run_finished: on_run_finished.set()
            input(f"Loaded CPG Index: {self.idx}. Press enter to start test next test")
            with on_run_start: on_run_start.notifyAll()

        # After 30 seconds, we progress to the next CPG
        if(self.dt0 > 5):
            self.idx += 1
            # If idx reached the end, quit
            if(self.idx == len(self.brains)):
                print("Test complete. Shutting down")
                exit()
            print("Loading next...")
            
            # Reset dt states
            self.spam_times = 20
            self.dt0 = 0
            self.capture_dt = False
            return

        self.brains[self.idx].control(dt, sensor_state, control_interface)

robot = ModularRobot(body=body,
                     brain=BatchTesterBrain()
)

# CAMERA IMPL
cameras_in_use = [
    cv2.VideoCapture(cameras[1]),
    cv2.VideoCapture(cameras[2])
]

def record_process():
    print("=== RECORDER IS ACTIVE ===")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (920, 920)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    write_to_file = cv2.VideoWriter(f"{time.strftime('%Y%m%d-%H%M%S')}.mp4", fourcc, 20.0, size)

    while True:
        with on_run_start:
            on_run_start.wait()
            write_to_file = cv2.VideoWriter(f"{time.strftime('%Y%m%d-%H%M%S')}.mp4", fourcc, 20.0, size)

        while True:
            if on_run_finished.is_set():
                print("=== Recording for this run finished, starting again ===")
                on_run_finished.clear()
                break
            
            # Read frames from both cameras
            ret1, frame1 = cameras_in_use[0].read()
            ret2, frame2 = cameras_in_use[1].read()
            
            if not ret1 or not ret2:
                print("Failed to grab frames.")
                break        
            
            frame1_resized = cv2.resize(frame1, size)
            frame2_resized = cv2.resize(frame2, size)

            # Concatenate frames side by side
            frame = cv2.hconcat([frame1_resized, frame2_resized])
            cv2.imshow("Stitched Feed", frame)
            write_to_file.write(frame)

    [c.release() for c in cameras_in_use]
    cv2.destroyAllWindows()

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

t = threading.Thread(target=record_process)

print("Initializing robot..")
t.start()
remote_control_with_polling_rate(
    config=config,
    port=20812,
    hostname="10.15.3.59",
    rate=10
)
t.join()