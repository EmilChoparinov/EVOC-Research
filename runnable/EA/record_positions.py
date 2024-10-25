"""
Goal:
Generates positional and parameter data into CSV's saved into the 
/out/record_positions directory. 

The positions are based on a randomly seeded CPG over the gecko_v2 model within 
a mujoco simulation.

WARNING!!! This script will CLEAR /out/record_positions directory and write new
files. If you wish to keep them, move them out.
"""

from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.experimentation.rng import make_rng, seed_from_time
from revolve2.modular_robot.body.v2 import BodyV2, BrickV2
body = gecko_v2()

# === Program variables ========================================================
num_itrs = 10                     # Iteration count (how many runs to produce)
time_max = 30                     # Time each simulation takes (in seconds)
sample_freq = 30                  # Sample rate (per second) in the sim 
def seed(): seed_from_time(False) # Seed to generate CPG from
def csv_map_f(_body: BodyV2): 
    return  {                       # Positions within `body` to record into CSV
        "head": _body.core_v2,
        "middle": _body.core_v2.back_face.bottom.attachment,
        "rear": _body.core_v2.back_face.bottom.attachment.front.attachment,
        "right_front": _body.core_v2.right_face.bottom.attachment,
        "left_front": _body.core_v2.left_face.bottom.attachment,
        "right_hind":_body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment,
        "left_hind": _body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment
    }

csv_map = csv_map_f(body)
csv_cols = ["frame_id", "head", "middle", "rear", "right_front", "left_front", "right_hind", "left_hind","center-euclidian"]
# ==============================================================================

from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.modular_robot_simulation import SceneSimulationState
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.standards import terrains
from revolve2.simulation.scene import UUIDKey
from revolve2.modular_robot import ModularRobot

import os
import glob

import pandas as pd
import numpy as np

# === Pure environment to perform simulation ========================
def perform_simulation(csv_id: int) -> tuple[list[SceneSimulationState], ModularRobot]:
    # If seed = seed_from_time(False) 
    #    then make_rng(seed) = make_rng_time_seed(False)
    cpg_param_init = make_rng(seed()) 


    brain = BrainCpgNetworkNeighborRandom(body, cpg_param_init)

    print("matrix")
    print(brain._weight_matrix)
    pd.DataFrame(np.array(brain._weight_matrix)).to_csv(f"./../out/record_positions/cpg-{csv_id}.csv", index=False, header=False)


    robot = ModularRobot(body, brain)

    scene = ModularRobotScene(terrains.flat())
    scene.add_robot(robot)

    simulator = LocalSimulator(headless=True)

    return (simulate_scenes(
        simulator=simulator,
        batch_parameters=make_standard_batch_parameters(
            simulation_time=time_max,
            sampling_frequency=sample_freq),
        scenes=scene
    ), robot)
# ==============================================================================

# === Record data into CSV =====================================================
import pdb
def record_into_csv(
    csv_id: int, 
    csv_cols: list[str],
    csv_map: dict[str, BrickV2],
    simulation: tuple[list[SceneSimulationState], ModularRobot]
    ):
    df = pd.DataFrame(columns=csv_cols)
    states = simulation[0]
    robot = simulation[1]
    
    for idx, state in enumerate(states):
        absolute_pos_f = state.get_modular_robot_simulation_state(robot).get_module_absolute_pose
        
        center_x = []
        center_y = []
        def map_col_to_value(col: str):
            match col:
                case "frame_id": return idx
                case "center-euclidian": return 0 # We calculate this later
                case _: 
                    abs_pos = absolute_pos_f(csv_map[col])
                    center_x.append(abs_pos.position.x)
                    center_y.append(abs_pos.position.z) 
                    return f"({abs_pos.position.x},{abs_pos.position.z})"
        row = {col: map_col_to_value(col) for col in csv_cols}
        row["center-euclidian"] = f"({np.array(center_x).mean()},{np.array(center_y).mean()})"
        df.loc[len(df.index)] = row
    df.to_csv(f"./run-{csv_id}.csv", index=False)
# ==============================================================================


# # Cleanup routine
# os.makedirs("../out/record_positions", exist_ok=True)
# file_list = glob.glob("../out/record_positions/*")
# for f in file_list: os.remove(f)

# for run_id in range(num_itrs):
#     record_into_csv(run_id, csv_cols, csv_map, perform_simulation(run_id))    
