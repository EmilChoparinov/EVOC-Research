"""
Goal:
Provide an example simulation of a customized robot performing a "snowflake" or
"swimming" motion.
"""

weight_matrix_csv_filepath = "./EA/best-cpg-gen-99.csv" 

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
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.standards.modular_robots_v2 import gecko_v2

import pandas as pd
import numpy as np

rng = make_rng_time_seed()

robot_body = gecko_v2()
robot_brain = BrainCpgNetworkNeighborRandom(robot_body, rng)
df = pd.read_csv(weight_matrix_csv_filepath, header=None)
robot_brain._weight_matrix = df.to_numpy(dtype=float)

robot = ModularRobot(robot_body, robot_brain)

def main() -> None:
    setup_logging()
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)
    
    simulate_scenes(
        simulator=LocalSimulator(),
        batch_parameters=make_standard_batch_parameters(simulation_time=60),
        scenes=scene
    )

if __name__ == "__main__": main()