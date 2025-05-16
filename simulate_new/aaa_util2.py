import math

import numpy as np
import pandas as pd

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, active_hinges_to_cpg_network_structure_neighbor, \
    CpgNetworkStructure
from revolve2.modular_robot.body.base import ActiveHinge

from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from simulate_new import stypes, data, evaluate


def visualize_individual(genotype):
    body_shape = gecko_v2()
    cpg_network_struct, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    robot = ModularRobot(body=body_shape, brain=BrainCpgNetworkStatic.uniform_from_params(
        params=genotype,
        cpg_network_structure=cpg_network_struct,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping))

    scenes = []
    s = ModularRobotScene(terrain=terrains.flat())
    s.add_robot(robot)
    scenes.append(s)

    simulate_scenes(
        simulator=LocalSimulator(headless=False, num_simulators=1, start_paused=True),
        batch_parameters=make_standard_batch_parameters(
            simulation_time=30,
            sampling_frequency=30
        ),
        scenes=scenes
    )

def evaluate_individual_distance(genotype):
    def simulate(solution_set: list[stypes.solution],
                           cpg_struct: CpgNetworkStructure,
                           body_shape: BodyV2, body_map: any):

        robots = [
            ModularRobot(
                body=body_shape,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=solution,
                    cpg_network_structure=cpg_struct,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=body_map))
            for solution in solution_set]

        def new_robot_scene(robot: ModularRobot) -> ModularRobotScene:
            s = ModularRobotScene(terrain=terrains.flat())
            s.add_robot(robot)
            return s

        scenes = [new_robot_scene(robot) for robot in robots]
        return (robots,
                simulate_scenes(
                    simulator=LocalSimulator(headless=True, num_simulators=1),
                    batch_parameters=make_standard_batch_parameters(
                        simulation_time=30,
                        sampling_frequency=30),
                    scenes=scenes))

    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    animal_data_file = "Files/slow_lerp_4.csv"
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
    state = stypes.EAState(
        generation=1,
        run=1,
        alpha=-1,
        animal_data=animal_data,
    )

    robot, behavior = simulate([genotype], cpg_struct, body_shape, mapping)
    df_behavior = data.behaviors_to_dataframes(robot, behavior, state, z_axis=False)

    f = -evaluate.evaluate_by_distance(df_behavior[0])
    return f

def evaluate_individual_similarity(genotype):
    def simulate(solution_set: list[stypes.solution],
                           cpg_struct: CpgNetworkStructure,
                           body_shape: BodyV2, body_map: any):

        robots = [
            ModularRobot(
                body=body_shape,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=solution,
                    cpg_network_structure=cpg_struct,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=body_map))
            for solution in solution_set]

        def new_robot_scene(robot: ModularRobot) -> ModularRobotScene:
            s = ModularRobotScene(terrain=terrains.flat())
            s.add_robot(robot)
            return s

        scenes = [new_robot_scene(robot) for robot in robots]
        return (robots,
                simulate_scenes(
                    simulator=LocalSimulator(headless=True, num_simulators=1),
                    batch_parameters=make_standard_batch_parameters(
                        simulation_time=30,
                        sampling_frequency=30),
                    scenes=scenes))

    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    animal_data_file = "Files/slow_lerp_4.csv"
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
    state = stypes.EAState(
        generation=1,
        run=1,
        alpha=-1,
        animal_data=animal_data
    )

    robot, behavior = simulate([genotype], cpg_struct, body_shape, mapping)
    df_behavior = data.behaviors_to_dataframes(robot, behavior, state, z_axis=False)

    f = evaluate.evaluate_by_4_angles(df_behavior[0], state.animal_data)
    return f

def optimize_individual(genotype, std_dev=0.005):
    best_fitness = evaluate_individual_distance(genotype)
    best_genotype = genotype
    ok = False
    while not ok:
        ok = True
        for _ in range(32):
            x = best_genotype.copy()
            for i in range(9):
                x[i] += np.random.normal(loc=0, scale=std_dev)
                x[i] = min(max(x[i], -2.5), 2.5)
            fitness = evaluate_individual_distance(x)
            if fitness > best_fitness:
                best_fitness = fitness
                best_genotype = x
                ok = False
                break
    return best_genotype