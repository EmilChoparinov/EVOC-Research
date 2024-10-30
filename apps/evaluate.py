"""
Goal:
The goal of this python file is to perform another evaluation over the specific
genotype requested `PARAMS`.
"""

# TODO: Script is broken. Please fix it!

from revolve2.simulators.mujoco_simulator import LocalSimulator

import numpy as np
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)

import math
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
from revolve2.standards import fitness_functions, terrains
from revolve2.standards.modular_robots_v2 import gecko_v2

PARAMS = np.array(
   [-2.99999447,  1.81308186, -0.0331997 ,  0.06852362,  2.89306811,
       -0.58302268,  2.76108739, -2.99385889, -1.80114772]
    # [-1.99028127,  0.08435464,  0.00391817, -0.02174904,  1.94509952,
    #     1.60096773,  1.61155222, -1.99356261, -1.12994603]
)

def main() -> None:
    setup_logging()

    body = gecko_v2()

    active_hinges = body.find_modules_of_type(ActiveHinge)

    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
    
    robot = ModularRobot(
                body=body,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=PARAMS,
                    cpg_network_structure=cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=output_mapping,
                ),
            )
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)
    
    simulate_scenes(
        simulator=LocalSimulator(),
        batch_parameters=make_standard_batch_parameters(),
        scenes=scene
    )



if __name__ == "__main__":
    main()
