import json
import logging
import numpy as np
from cma import CMAOptions, CMAEvolutionStrategy

from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.standards import terrains
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
import math
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from simulate_new.stypes import objective_type
import os
import simulate_new.stypes as stypes
import simulate_new.evaluate as evaluate
import simulate_new.data as data
import pandas as pd
from pathlib import Path

def create_state(
        generation: int, run: int, alpha: float, animal_data: pd.DataFrame):
    return stypes.EAState(
        generation=generation, run=run, alpha=alpha, animal_data=animal_data)

def create_config(ttl: int, freq: int):
    return stypes.EAConfig(ttl=ttl, freq=freq)

def local_path(path: str, module: str = "Outputs") -> str:
    full_path =\
        os.path.join(os.path.dirname(os.path.abspath(__file__)), module, path)

    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    return full_path

def file_idempotent(state: stypes.EAState, objective: objective_type) -> str:
    return f"./run-{state.run}-alpha-{state.alpha}-{objective}.csv"


def simulate_simple(solution: stypes.solution):
    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))
    return simulate_solutions(
        [solution], cpg_struct, body_shape, create_config(30, 30))

def simulate_solutions(solution_set: list[stypes.solution],
                       cpg_struct: CpgNetworkStructure,
                       body_shape: BodyV2, body_map: any,
                       config: stypes.EAConfig):

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
                simulator=LocalSimulator(headless=True, num_simulators=8),
                batch_parameters=make_standard_batch_parameters(
                    simulation_time=config.ttl,
                    sampling_frequency=config.freq),
                scenes=scenes))

def optimize(state: stypes.EAState, config: stypes.EAConfig, objective: objective_type):
    NUMBER_OF_GENES = 9
    POP_SIZE = 16
    NGEN = state.generation

    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    best_over_all_score = math.inf
    best_over_all_sol = None

    def evaluate_population(individuals):
        robots, behaviors = simulate_solutions(individuals, cpg_struct, body_shape, mapping, config)
        df_behaviors = data.behaviors_to_dataframes(robots, behaviors, state, z_axis=True)

        match objective:
            case "Distance":
                return [evaluate.evaluate_by_distance(df) for df in df_behaviors]
            case "MSE":
                return [evaluate.evaluate_by_mse(df, state.animal_data) for df in df_behaviors]
            case "DTW":
                return [evaluate.evaluate_by_dtw(df, state.animal_data) for df in df_behaviors]
            case "2_Angles":
                return [evaluate.evaluate_by_2_angles(df, state.animal_data) for df in df_behaviors]
            case "4_Angles":
                return [evaluate.evaluate_by_4_angles(df, state.animal_data) for df in df_behaviors]
            case "All_Angles":
                return [evaluate.evaluate_by_all_angles(df, state.animal_data) for df in df_behaviors]
            case "Work":
                return [evaluate.evaluate_mechanical_work(df) for df in df_behaviors]

    cma_es_options = CMAOptions()
    cma_es_options.set("bounds", [-2.5, 2.5])
    cma_es_options.set("popsize", POP_SIZE)

    cma_es = CMAEvolutionStrategy(NUMBER_OF_GENES * [0.0], 0.5, cma_es_options)

    to_dump = []
    for gen in range(NGEN):
        logging.info(f"Run {state.run} - Generation {gen + 1}/{NGEN}")

        population = cma_es.ask()
        population_list = [ind.tolist() for ind in population]
        fitnesses = evaluate_population(population_list)
        cma_es.tell(population, fitnesses)

        # Save best solution
        best_idx = np.argmin(fitnesses)
        best_score = fitnesses[best_idx]
        if best_score < best_over_all_score:
            best_over_all_score = best_score
            best_over_all_sol = population_list[best_idx]

        logging.info(f"Best distance: {best_over_all_score}")
        logging.info(f"Best sol: {best_over_all_sol}")

        robot, behavior = simulate_solutions([best_over_all_sol], cpg_struct, body_shape, mapping, config)
        df = data.behaviors_to_dataframes(robot, behavior, state, z_axis=False)[0]
        new_entry = {
            "generation": gen,
            "genotype": best_over_all_sol,
            "distance": -evaluate.evaluate_by_distance(df),
            "MSE": evaluate.evaluate_by_mse(df, state.animal_data),
            "DTW": evaluate.evaluate_by_dtw(df, state.animal_data),
            "2_Angles": evaluate.evaluate_by_2_angles(df, state.animal_data),
            "4_Angles": evaluate.evaluate_by_4_angles(df, state.animal_data),
            "All_Angles": evaluate.evaluate_by_all_angles(df, state.animal_data),
            "Work": evaluate.evaluate_mechanical_work(df),
        }
        to_dump.append(new_entry)

    os.makedirs("Outputs/CMAES_CSVs", exist_ok=True)
    with open(f"Outputs/CMAES_CSVs/best_solutions_run_{state.run}.json", "w") as file:
        json.dump(to_dump, file, indent=2)


    logging.info(f"Best {objective}: {best_over_all_score}")
    logging.info(f"Finished run: {state.run}")