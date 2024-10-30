from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.standards import terrains

from revolve2.experimentation.rng import seed_from_time

from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor

from revolve2.simulators.mujoco_simulator import LocalSimulator

import cma
import pandas as pd

# Setup Routines ===============================================================
def generate_cma() -> cma.CMAEvolutionStrategy:
    # Generate the initial state. The list of 0's is as long as the genotype
    # (the cpg network parameter length)
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

# Gecko Parameters =============================================================
body_shape = gecko_v2()

# Generate a CPG based on the gecko_v2's hinge tree
cpg_network_struct, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
    body_shape.find_modules_of_type(ActiveHinge)
)

# Experiment Parameters ========================================================
concurrent_simulators = 8
ea_runs_cnt = 5
ea_generations_cnt = 500

# Simulation Parameters ========================================================
simulator = LocalSimulator(headless=True, num_simulators=concurrent_simulators)
terrain = terrains.flat()

# The living time of the simulation in seconds
simulation_ttl = 30

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