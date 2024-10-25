import pandas as pd


import math

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import fitness_functions, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from record_positions import csv_map_f, record_into_csv


class Evaluator:
    _simulator: LocalSimulator
    _terrain: Terrain
    _cpg_network_structure: CpgNetworkStructure
    _body: Body
    _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        cpg_network_structure: CpgNetworkStructure,
        body: Body,
        output_mapping: list[tuple[int, ActiveHinge]],
    ) -> None:
        """
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()
        self._cpg_network_structure = cpg_network_structure
        self._body = body
        self._output_mapping = output_mapping

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
        df: pd.DataFrame,
        iteration_idx: int
    ) -> npt.NDArray[np.float_]:
        
        # Create robots from the brain parameters.
        robots = [
            ModularRobot(
                body=self._body,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=params,
                    cpg_network_structure=self._cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=self._output_mapping,
                ),
            )
            for params in solutions
        ]

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(
                simulation_time=30,
                sampling_frequency=30),
            scenes=scenes,
        )
        
        # Calculate the xy displacements.
        xy_displacements = [
            fitness_functions.xy_displacement(
                states[0].get_modular_robot_simulation_state(robot),
                states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot, states in zip(robots, scene_states)
        ]
        
        fittest_idx, most_displacement = max(
            enumerate(xy_displacements), 
            key=lambda x: x[1]
        )
        
        fittest_robot = robots[fittest_idx]
        csv_map = csv_map_f(fittest_robot.body)
        
        fittest_scene = scene_states[fittest_idx]

        # Record state into CSV.       
        for idx, state in enumerate(fittest_scene):
            absolute_pose_f = state.get_modular_robot_simulation_state(fittest_robot).get_module_absolute_pose

            center_x = []
            center_y = []
            def col_map(col: str):
                match col:
                    case "generation_id": return iteration_idx
                    case "center-euclidian": return 0 # calculate this after
                    case "generation_best_fitness_score": return most_displacement
                    case "frame_id": return idx
                    case _:
                        abs_pose = absolute_pose_f(csv_map[col])
                        center_x.append(abs_pose.position.x)
                        center_y.append(abs_pose.position.z)
                        return f"({abs_pose.position.x},{abs_pose.position.z})"
            row = {col: col_map(col) for col in df.columns.tolist()}
            row["center-euclidian"] = f"({np.array(center_x).mean()},{np.array(center_y).mean()})"
            df.loc[len(df.index)] = row
            
        # Record the winning cpg
        brain: BrainCpgNetworkStatic = robot.brain
        pd.DataFrame(np.array(brain._weight_matrix)).to_csv(f"best-cpg-gen-{iteration_idx}.csv", index=False, header=False)
    
    
        return np.array(xy_displacements)
