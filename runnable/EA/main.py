"""Main script for the example."""

import logging

import cma
import config
import pandas as pd
from evaluator import Evaluator

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import seed_from_time
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)


def main() -> None:
    setup_logging()

    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)

    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        cpg_network_structure=cpg_network_structure,
        body=config.BODY,
        output_mapping=output_mapping,
    )

    # initial_mean = cpg_network_structure.num_connections * [0.5]
    initial_mean = cpg_network_structure.num_connections * [0.0]
    
    # Init the cma optimizer.
    options = cma.CMAOptions()
    options.set("bounds", [-2.0, 2.0])
    
    rng_seed = seed_from_time() % 2**32  # Cma seed must be smaller than 2**32.
    options.set("seed", rng_seed)
    opt = cma.CMAEvolutionStrategy(initial_mean, config.INITIAL_STD, options)

    generation_index = 0

    logging.info("Start optimization process.")

    # CSV generation headers
    csv_cols = [
        "generation_id", "generation_best_fitness_score", "frame_id",  "head", 
        "middle", "rear", "right_front", 
        "left_front", "right_hind", 
        "left_hind","center-euclidian"
    ]
    df = pd.DataFrame(columns=csv_cols)

    while generation_index < config.NUM_GENERATIONS:
        logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")

        solutions = opt.ask()
        fitnesses = -evaluator.evaluate(solutions, df, generation_index)
        opt.tell(solutions, fitnesses)
        
        logging.info(f"{opt.result.xbest=} {opt.result.fbest=}")
        generation_index += 1

    # TODO: MAKE THIS PRINT IN THE WHILE!! We don't want the situation such that
    #       we train for 5 hours and we lose all data.
    df.to_csv(f"./generation_run.csv", index=False)

if __name__ == "__main__":
    main()
