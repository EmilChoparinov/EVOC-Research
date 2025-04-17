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

import simulate.data as data
import simulate.stypes as stypes 
import simulate.ea as ea
import simulate.data as data
import simulate.evaluate as evaluate
import typing

parser = argparse.ArgumentParser()
process_pd = lambda csv: data.convert_tuple_columns(pd.read_csv(csv))

parser.add_argument(
    "--cpg", type=str,
    help="The Genotype to generate data from", required=True)

parser.add_argument(
    "--animal-data", type=process_pd, 
    help=".csv file containing animal data", required=True)

parser.add_argument(
    "--alpha", type=float, 
    required=True, help="Alpha value between 0 and 1.")

parser.add_argument(
    "--similarity-type", type=str, required=True,choices=list(stypes.similarity_type.__args__), 
    help="Specify the fitness function for EA.")

parser.add_argument(
    "--runs", type=int, 
    default=5, help="Times to run EA")

parser.add_argument(
    "--ttl", type=int,
    default=30, 
    help="The time to live (TTL) in seconds of the simulation. How long the \
        simulation will go for")

parser.add_argument("--gens", type=int, default=300, help="How many generations per run")

class Args(typing.NamedTuple):
    animal_data: pd.DataFrame
    alpha: float
    similarity_type: stypes.similarity_type
    gens: int
    runs: int
    ttl: int
    cpg: str

args: Args = parser.parse_args()

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
    generation=-1, 
    run=-1, 
    alpha=args.alpha,
    similarity_type=args.similarity_type,
    animal_data=args.animal_data)

dfs = data.behaviors_to_dataframes(robots, behaviors, psuedo_state)

scores = evaluate.evaluate(dfs, psuedo_state, -1)
score, df = evaluate.most_fit(scores, dfs)

import pdb; pdb.set_trace()

data.apply_statistics(df, score, psuedo_state, -1)
df.to_csv(ea.file_idempotent(psuedo_state), index=False)
print("Complete")