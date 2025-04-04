import math
import random
import pandas as pd
import logging
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

from revolve2.standards import terrains
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, active_hinges_to_cpg_network_structure_neighbor, \
    CpgNetworkStructure
from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

import simulate.stypes as stypes
import simulate.data as data
import simulate.evaluate as evaluate


def nsga2_optimize(state: stypes.EAState, config: stypes.EAConfig):
    NUMBER_OF_GENES = 9
    POP_SIZE = 32
    NGEN = state.generation

    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge)
    )

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0)) # max distance, min similarity
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -2.5, 2.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NUMBER_OF_GENES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def simulate(solution_set: list[stypes.solution],
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

    def evaluate_population(individuals):
        robots, behaviors = simulate(individuals, cpg_struct, body_shape, mapping, config)
        df_behaviors = data.behaviors_to_dataframes(robots, behaviors, state)

        distances = [-evaluate.evaluate_by_distance(behavior) for behavior in df_behaviors]
        similarities = [evaluate.evaluate_by_4_angles(behavior, state.animal_data) for behavior in df_behaviors]

        return list(zip(distances, similarities))

    def evaluate_ind(individual):
        return 0.0, 0.0

    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()

    # We calculate the initial fitness of the population
    fitnesses = evaluate_population(pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Start the learning process
    for gen in range(NGEN):
        logging.info(f"[NSGA2] Generation {gen+1}/{NGEN}")

        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.2)

        # We calculate only the offsprings fitness
        fitnesses = evaluate_population(offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(offspring + pop, k=POP_SIZE)
        hof.update(pop)

        # We save the results
        df_scores = pd.DataFrame(fitnesses, columns=["objective_1_distance", "objective_2_similarity"])
        df_scores["objective_1_distance"] = [fit[0] for fit in fitnesses]
        df_scores["objective_2_similarity"] = [fit[1] for fit in fitnesses]
        df_scores["generation"] = gen

        filename = f"nsga2-gen-{gen}-pareto.csv"
        df_scores.to_csv(filename, index=False)
        logging.info(f"[NSGA2] Saved Pareto front for gen {gen} to {filename}")

    # Saving the final results:
    final_df = pd.DataFrame([ind for ind in hof])
    final_df.to_csv("nsga2-final-pareto.csv", index=False)
    logging.info("[NSGA2] Optimization complete.")

    # Showing the final results:
    pareto_fitnesses = [ind.fitness.values for ind in hof]
    objective_1 = [fit[0] for fit in pareto_fitnesses]
    objective_2 = [fit[1] for fit in pareto_fitnesses]

    plt.figure(figsize=(8, 6))
    plt.scatter(objective_1, objective_2, color='blue', label='Pareto Front')
    plt.xlabel('Objective 1: Distance (maximize)')
    plt.ylabel('Objective 2: Similarity (minimize)')
    plt.title('NSGA-II Pareto Front')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("nsga2_pareto_plot.png")
    plt.show()
