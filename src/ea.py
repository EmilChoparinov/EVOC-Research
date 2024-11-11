from revolve2.experimentation.logging import setup_logging

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic

from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

import logging
import math
import numpy as np
from typedef import simulated_behavior, genotype
from data_collection import record_elite_generations

import data_collection
import evaluate
import config
import csv
import random
import os

def export_ea_metadata(run_id: int = 0):
    seed = config.random_seed if hasattr(config, 'random_seed') else random.randint(0, 1000000)
    initial_mean = config.initial_mean if hasattr(config, 'initial_mean') else [0.0] * 9
    initial_sigma = config.initial_sigma if hasattr(config, 'initial_sigma') else 0.5
    bounds = config.bounds if hasattr(config, 'bounds') else (-2, 2)

    metadata_file = "ea-run-metadata.csv"

    metadata = [
        ["Run ID", run_id],
        ["Seed", seed],
        ["Initial Mean", initial_mean],
        ["Initial Sigma", initial_sigma],
        ["Bounds", bounds]
    ]

    with open(metadata_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Parameter", "Value"])
        writer.writerows(metadata)

    logging.info(f"Metadata for run {run_id} exported to {metadata_file}")


from rotation_scaling import get_data_with_forward_center,translation_rotation


# TODO: This function has been primed to be embarrassingly parallel if more performance required.
# def process_ea_iteration(max_gen: int, max_runs: int = config.ea_runs_cnt):
def process_ea_iteration(max_gen: int, max_runs: int = config.ea_runs_cnt, fitness_function: str = "distance",
                             alpha: float = 1.0):

    if(max_runs == 0): return

    # record data
    export_ea_metadata(max_runs)

    # Stack `max_run` times this function and save output
    process_ea_iteration(max_gen, max_runs - 1)
    
    setup_logging(file_name=config.generate_log_file(max_runs))
    logging.info("Start CMA-ES Optimization")
    
    cma_es = config.generate_cma()
    
    # Write the columns into the csv 
    behavior_csv = config.generate_fittest_xy_csv(max_runs)
    config.write_buffer.to_csv(behavior_csv, index=False)


    # EA Loop
    for generation_i in range(max_gen):
        logging.info(
            f"Performing Generation {generation_i} / {max_gen}")

        # Evaluation Step
        solutions = cma_es.ask()

        robots, behaviors = ea_simulate_step(solutions)
        #Rotation
        translation_rotation(get_data_with_forward_center(robots, behaviors))
        # TODO scaling
        
        #fitnesses = -evaluate.evaluate(robots, behaviors)
        fitnesses = None

        if fitness_function == "distance":
            fitnesses = -evaluate.evaluate_distance(robots, behaviors)
        elif fitness_function == "similarity":
            fitnesses = -evaluate.evaluate_similarity(robots, behaviors)
        elif fitness_function == "blended":
            # Calculate the weighted average of the two adaptations, weights determined by alpha
            distance_fitness = evaluate.evaluate_distance(robots, behaviors)
            similarity_fitness = evaluate.evaluate_similarity(robots, behaviors)
            fitnesses = -((alpha * distance_fitness) + ((1 - alpha) * similarity_fitness))

        if fitnesses is None:
            raise ValueError(
                "Fitnesses could not be calculated. Please check the fitness_function value and corresponding evaluation functions.")

        cma_es.tell(solutions, fitnesses)

        # Find the most fit individual of this generation
        best_robot, best_behavior, best_fitness = evaluate.find_most_fit(fitnesses, robots, behaviors)

        # Data Collection Step
        data_collection.record_behavior(
            best_robot, best_fitness, best_behavior, generation_id=generation_i, alpha=alpha,
            fitness_function=fitness_function)

        data_collection.record_elite_generations(
            run_id=max_runs, generation=generation_i, fitness=best_fitness, matrix=best_robot.brain._weight_matrix,
            alpha=alpha, fitness_function=fitness_function)

        # top 3 fitness and corresponding robots and weight matrices
        top_3_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:3]
        top_3_list = [(robots[i], fitnesses[i], robots[i].brain._weight_matrix) for i in top_3_indices]

        for i, (robot, fitness, weights_matrix) in enumerate(top_3_list):
            record_elite_generations(run_id=max_runs, generation=generation_i, fitness=fitness, matrix=weights_matrix)

        logging.info(f"{cma_es.result.xbest=}\n{cma_es.result.fbest=}")

        # Record CPG configuration at the end of the EA run
        if(generation_i + 1 == max_gen):
            data_collection.record_cpg(best_robot, max_runs)
        
        # Clean-up Step. Flush buffer to disk after every evaluation step
        logging.info(f"Recording best fit behavior to {behavior_csv}")
        config.write_buffer.to_csv(behavior_csv, index=False, header=False, mode='a')
        config.write_buffer.drop(config.write_buffer.index, inplace=True)
    
    # Do not need to flush the buffer at this step because it's always the
    # last thing the loop does.
    logging.info(f"EA Iteration {max_runs} complete")
    
def ea_simulate_step(solution_set: list[genotype]) -> tuple[list[ModularRobot], list[simulated_behavior]]:
    """
    Perform one simulation step in the EA. An item from `solution_set` is mapped
    to a list of simulation states produced by the simulator. These states 
    contain Pose data. 

    Returns a list of the behaviors alongside the robots
    """
    # Construct the robots
    robots = [
        ModularRobot(
            body=config.body_shape,
            brain=BrainCpgNetworkStatic.uniform_from_params(
                params=solution,
                cpg_network_structure=config.cpg_network_struct,
                initial_state_uniform=math.sqrt(2) * 0.5, 
                output_mapping=config.output_mapping
            )
        )
        for solution in solution_set
    ]
    
    def new_robot_scene(robot: ModularRobot) -> ModularRobotScene:
        s = ModularRobotScene(terrain=config.terrain)
        s.add_robot(robot)
        return s
    
    scenes = [new_robot_scene(robot) for robot in robots]
    
    # Simulate all constructed scenes
    return (robots, simulate_scenes(
        simulator=config.simulator,
        batch_parameters=make_standard_batch_parameters(
            simulation_time=config.simulation_ttl,
            sampling_frequency=config.collection_rate
        ),
        scenes=scenes
    ))
    