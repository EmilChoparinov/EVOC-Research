from revolve2.experimentation.logging import setup_logging

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic

from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from config import alpha
import logging
import math
import numpy as np

from VAE import plot_fitness,infer_on_csv,save_to_csv
from typedef import simulated_behavior, genotype
from data_collection import record_elite_generations,record_best_fitness_generation_csv
from animal_similarity import create_simulation_video
import data_collection
import evaluate
import config
import csv
import random
import os
import pandas as pd

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
def process_ea_iteration(alpha,fitness_function,similarity_type,max_gen: int, max_runs: int = config.ea_runs_cnt):

    alpha = alpha if alpha is not None else config.alpha

    if(max_runs == 0): return

    # record data
    export_ea_metadata(max_runs)

    # Stack `max_run` times this function and save output
    process_ea_iteration(alpha,fitness_function,similarity_type,max_gen, max_runs - 1)
    
    setup_logging(file_name=config.generate_log_file(max_runs))
    logging.info("Start CMA-ES Optimization")
    
    cma_es = config.generate_cma()

    # Read animal data
    df_animal = pd.read_csv('./src/model/slow_with_linear_4.csv')

    df_animal = infer_on_csv(df_animal)
    distance_all=[]
    animal_similarities_all=[]
    fitnesses_all = []
    # similarity_type = "DTW"
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
        fitnesses,distance,animal_similarity= evaluate.evaluate(robots, behaviors,df_animal,alpha,similarity_type )

        distance_all.append(distance)
        animal_similarities_all.append(animal_similarity)
        fitnesses_all.append(fitnesses)


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

        record_best_fitness_generation_csv(max_runs,best_robot, best_fitness, best_behavior, generation_id=generation_i, alpha=alpha,
            fitness_function=fitness_function,similarity_type=similarity_type)

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
        try:
            config.write_buffer.to_csv(behavior_csv, index=False, header=False, mode='a')
            config.write_buffer.drop(config.write_buffer.index, inplace=True)
        except Exception as e:
            logging.error(f"Error while writing to CSV: {e}")
            raise
    save_to_csv(max_runs,fitnesses_all,distance_all, animal_similarities_all,alpha,similarity_type)

    create_simulation_video(max_runs,alpha,similarity_type)
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
    