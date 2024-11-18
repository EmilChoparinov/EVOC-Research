"""
Goal: The goal of this file is to store all configurations for the source. 
      Any parameters with interest that could be manually altered should be
      stored in this file.
      
Properties:

initial_mean: Generates the initial state for CMA-ES. As long as the genotype.
initial_std:  The standard deviation to start CMS-ES with.
bounds:       CMA-ES bounding property
population:   The amount of robots in the simulation

body_to_csv_map: Generates a dictionary relating the body parts of the robot 
                 semantically with its object in the simulation

generated_fittest_xy_csv: Generates a filename string for the fittest per 
                          generation xy plot

generate_log_file: Generates the filename string for the simulation log output

body_shape:         The fixed morphology to use in the simulation
cpg_network_struct: The CPG network generated based on `body_shape` morphology
output_mapping:     A map between `body_shape` and the CPG

concurrent_simulators: The amount of simulations that should run concurrently
ea_runs_cnt:           The amount of runs the evolutionary algorithm should do
ea_generations_cnt:    The amount of generations to perform
generate_cma:          Generates CMA-ES object to perform EA run with

simulator:             Revolve2 simulator object
terrain:               Terrain to use in the simulation
simulation_ttl:        How long the simulation will last in sec (time to live)

csv_cols:              The columns used for `generate_fittest_xy_csv`
collection_rate:       How many times per second do we sample Pose data

write_buffer:          A pandas buffer object for `generate_fittest_xy_csv`

PhysMap:          DataClass containing the physical PIN configuration.
PhysMap.get_box:  Returns the ActiveHingeV2 object based on the body part given
PhysMap.map_with: Creates a map with the hinge object alongside the labels
"""

from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.standards import terrains

from revolve2.experimentation.rng import seed_from_time

from revolve2.modular_robot.body.v2 import BodyV2, ActiveHingeV2
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
from revolve2.modular_robot_physical import Config, UUIDKey

from revolve2.simulators.mujoco_simulator import LocalSimulator
from typing import Callable, Literal, TypedDict

import cma
import pandas as pd
import math
from dataclasses import dataclass

def generate_cma() -> cma.CMAEvolutionStrategy:
    initial_mean = cpg_network_struct.num_connections * [0.0]
    initial_std = 0.5
    bounds = [-2.5, 2.5]
    population = 10
    
    # CMA seed is constrained to be smaller than 2**32
    options = cma.CMAOptions()
    options.set("bounds", bounds)
    options.set("seed", seed_from_time() % 2 ** 32)
    # options.set("popsize", population)
    
    return cma.CMAEvolutionStrategy(initial_mean, initial_std, options)


def body_to_csv_map(body: BodyV2): 
    return  {
        "head": body.core_v2,
        "middle": body.core_v2.back_face.bottom.attachment,
        "rear": body.core_v2.back_face.bottom.attachment.front.attachment,
        "right_front": body.core_v2.right_face.bottom.attachment,
        "left_front": body.core_v2.left_face.bottom.attachment,
        "right_hind":body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment,
        "left_hind": body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment
    }

def generate_fittest_xy_csv(run_id: int = 0) -> str:
    return f"most-fit-xy-run-{run_id}.csv"

def generate_log_file(run_id: int = 0) -> str:
    return f"log-run-{run_id}.txt"

body_shape = gecko_v2()

# Generate a CPG based on the gecko_v2's hinge tree
cpg_network_struct, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
    body_shape.find_modules_of_type(ActiveHinge)
)

# Experiment Parameters ========================================================
concurrent_simulators = 8
ea_runs_cnt = 13
target_dist_per_run = [2.0]
target_angle_per_run = [math.radians(90)]
# target_angle_per_run = [math.radians(angle) for angle in range(0, 361, 30)]
ea_generations_cnt = 500

# Simulation Parameters ========================================================
simulator = LocalSimulator(headless=True, num_simulators=concurrent_simulators)
terrain = terrains.flat()

# The living time of the simulation in seconds
simulation_ttl = 10

# Data Collection Parameters ===================================================
csv_cols = [
        "generation_id", "generation_best_fitness_score", "frame_id", "head", 
        "middle", "rear", "right_front", "left_front", "right_hind", 
        "left_hind","center-euclidian"
    ]

# How many times per second do we sample the Pose of the robot
collection_rate = 30

# Every generation produced fills this DataFrame. At the end of each loop,
# it's contents are recorded to most-fit-xy-run-[X].csv
write_buffer = pd.DataFrame(columns=csv_cols)

# Write to this CSV the best genotype per EA run
best_solution_per_ea = "best-solutions-per-ea.csv"


# Typedef for valid box on robot
phys_box_names = Literal["left_arm", "right_arm", "torso", "tail", "left_leg", "right_leg"]

phys_config = dict[Literal["pin", "box", "is_inverse"]]

class PhysMap(TypedDict):
    pin: int
    extract: Callable[[BodyV2], ActiveHingeV2]

    def get_box(body: BodyV2,box: phys_box_names):
        phy_map: dict[phys_box_names, ActiveHingeV2] = {
            "left_arm": body.core_v2.left_face.bottom.attachment,
            "left_leg": body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment,
            "torso": body.core_v2.back_face.bottom.attachment,
            "right_arm": body.core_v2.right_face.bottom.attachment,
            "right_leg":body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment,
            "tail": body.core_v2.back_face.bottom.attachment.front.attachment,
        }
        return phy_map[box]
    
    def get_hinge(body: BodyV2,hinge: phys_box_names):
        phy_map: dict[phys_box_names, ActiveHingeV2] = {
            "left_arm": body.core_v2.left_face.bottom,
            "left_leg": body.core_v2.back_face.bottom.attachment.front.attachment.left,
            "torso": body.core_v2.back_face.bottom,
            "right_arm": body.core_v2.right_face.bottom,
            "right_leg":body.core_v2.back_face.bottom.attachment.front.attachment.right,
            "tail": body.core_v2.back_face.bottom.attachment.front,
        }
        return phy_map[hinge]
    

    def map_with(body: BodyV2) -> dict[phys_box_names, 'PhysMap']:
        return {
            "left_arm": {
                "pin": 0,
                "box": PhysMap.get_box(body, "left_arm"),
                "hinge": PhysMap.get_hinge(body, "left_arm"),
                "is_inverse": True
            },
            "left_leg": {
                "pin": 1,
                "box": PhysMap.get_box(body, "right_arm"),
                "hinge": PhysMap.get_hinge(body, "right_arm"),
                "is_inverse": False
            },
            "torso": {
                "pin": 8,
                "box": PhysMap.get_box(body, "torso"),
                "hinge": PhysMap.get_hinge(body, "torso"),
                "is_inverse": False
            },
            "right_arm": {
                "pin": 31,
                "box": PhysMap.get_box(body, "right_arm"),
                "hinge": PhysMap.get_hinge(body, "right_arm"),
                "is_inverse": False
            },
            "right_leg": {
                "pin": 30,
                "box": PhysMap.get_box(body, "right_leg"),
                "hinge": PhysMap.get_hinge(body, "right_leg"),
                "is_inverse": True
            },
            "tail": {
                "pin": 24,
                "box": PhysMap.get_box(body, "tail"),
                "hinge": PhysMap.get_hinge(body, "tail"),
                "is_inverse": False
            },
        }
 