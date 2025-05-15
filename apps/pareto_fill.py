import argparse
import pandas as pd

from datetime import datetime
import os
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
import ast
import copy

import simulate.data as data
import simulate.stypes as stypes 
import simulate.ea as ea
import simulate.data as data
import simulate.evaluate as evaluate
import typing
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Args(typing.NamedTuple):
    animal_data: pd.DataFrame
    alpha: float
    similarity_type: stypes.similarity_type
    gens: int
    runs: int
    ttl: int
    cpg: str

def simulate_fill(args):
    cpg = ast.literal_eval(args.cpg)
    body = gecko_v2()

    net_struct, mapping =\
        active_hinges_to_cpg_network_structure_neighbor(body.find_modules_of_type(ActiveHinge))

    robots, behaviors  = ea.simulate_solutions(
        solution_set=[cpg],
        cpg_struct=net_struct,
        body_map=mapping,
        body_shape=body,
        config=ea.create_config(ttl=args.ttl, freq=30)
    )

    robot = robots[0]
    behavior = behaviors[0]

    psuedo_state = ea.create_state(
        generation=0, 
        run=-1, 
        alpha=args.alpha,
        similarity_type=args.similarity_type,
        animal_data=args.animal_data)

    dfs = data.behaviors_to_dataframes(robots, behaviors, psuedo_state)

    scores = evaluate.evaluate(dfs, psuedo_state, -1)
    score, df = evaluate.most_fit(scores, dfs)

    data.apply_statistics(df, score, psuedo_state, -1)
    # df.to_csv(ea.file_idempotent(psuedo_state), index=False)
    # data.create_video_state(psuedo_state)
    print(f"Completed simulating genotype: {cpg}")
    return score, df

import os
import ast
import numpy as np
from datetime import datetime

def produce_angle_plot(df, animal, output_dir="."):
    def calculate_angle(p1, p2, p3):
        P = np.array([ast.literal_eval(p1),
                      ast.literal_eval(p2),
                      ast.literal_eval(p3)])
        vec1 = P[0] - P[1]
        vec2 = P[2] - P[1]
        ang1 = np.arctan2(vec1[1], vec1[0])
        ang2 = np.arctan2(vec2[1], vec2[0])
        return np.degrees(abs(ang2 - ang1))

    # Precompute animal angles once
    animal = animal.copy()
    animal["angle1"] = animal.apply(lambda r: calculate_angle(
        r["right_front"], r["left_hind"], r["left_front"]), axis=1)
    animal["angle2"] = animal.apply(lambda r: calculate_angle(
        r["left_front"], r["right_hind"], r["right_front"]), axis=1)

    # Select target generation
    target_gen = df['generation'].max()
    # df_prev = df.query(f"generation == {target_gen - 1}")
    df_prev = df

    # Compute robot angles
    robot_angles = df_prev.apply(lambda r: calculate_angle(
        r["right_front"], r["left_hind"], r["left_front"]), axis=1)

    # Time axis
    t_axis = np.arange(len(df_prev))
    animal_slice = animal["angle1"].iloc[:len(df_prev)]

    # Create figure+axes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_axis, robot_angles, label="Robot Angle 1 – RF LB LF")
    ax.plot(t_axis, animal_slice, label="Animal Angle 1 – RF LB LF")
    ax.set_title(f"Generation {target_gen} Robot vs Animal Angle 1 Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Angle (°)")
    ax.legend()
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"angle1_gen{target_gen}_{timestamp}.png"
    path = os.path.join(output_dir, fname)

    # Save and close
    fig.savefig(path)
    plt.close(fig)

    return path

# pareto_scores_230
# /home/cowboy/Research/EVOC-Research/CSVs_FINAL_2/pareto_scores_224.csv
if __name__ == '__main__':
    df = pd.read_csv("CSVs_FINAL_2/pareto_scores_165.csv")
    animal_df = pd.read_csv("./simulate/model/slow_with_linear_4.csv")

    def distance_col(genotype):        
        args: Args = Args(
            animal_data=data.convert_tuple_columns(pd.read_csv("simulate/model/slow_with_linear_4.csv")),
            alpha=0.5   ,
            similarity_type="distance",
            cpg=genotype['genotype'],
            gens=1,
            runs=1,
            ttl=30
        )
        results = simulate_fill(args)
        produce_angle_plot(results[1], animal_df)
        return results[0]['data_distance']

    df['distance_synth'] = df.apply(distance_col, axis=1)
    # df.to_csv('pareto_fill.csv')