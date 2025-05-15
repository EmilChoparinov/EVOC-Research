import json
import math
import random
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
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
from mpl_toolkits.mplot3d import Axes3D

def nsga2_optimize(state: stypes.EAState, config: stypes.EAConfig):
    NUMBER_OF_GENES = 9
    POP_SIZE = 12
    NGEN = state.generation

    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    distances = deque(maxlen=POP_SIZE * 4)
    symmetries = deque(maxlen=POP_SIZE * 4)
    mean_dist = 0
    std_dist = 1
    mean_sim = 0
    std_sim = 1

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

        work = [evaluate.evaluate_mechanical_work(behavior, state.animal_data) for behavior in df_behaviors]
        distances = [(-evaluate.evaluate_by_distance(behavior) - mean_dist) / std_dist for behavior in df_behaviors]
        similarities = [(evaluate.evaluate_by_4_angles(behavior, state.animal_data) - mean_sim) / std_sim for behavior in df_behaviors]
        #nr_of_bad_frames = [evaluate.evaluate_nr_of_bad_frames(df) for df in df_behaviors]

        # If you want to use the number of bad frames you just have to return list(zip(distances, nr_of_bad_frames)) - Everything should stay the same
        return list(zip(work, distances, similarities))
    
    def evaluate_population_2obj(individuals):
        robots, behaviors = simulate(individuals, cpg_struct, body_shape, mapping, config)
        df_behaviors = data.behaviors_to_dataframes(robots, behaviors, state)

        work = [evaluate.evaluate_mechanical_work(behavior, state.animal_data) for behavior in df_behaviors]
        # distances = [(-evaluate.evaluate_by_distance(behavior) - mean_dist) / std_dist for behavior in df_behaviors]
        similarities = [(evaluate.evaluate_by_4_angles(behavior, state.animal_data) - mean_sim) / std_sim for behavior in df_behaviors]
        #nr_of_bad_frames = [evaluate.evaluate_nr_of_bad_frames(df) for df in df_behaviors]

        # If you want to use the number of bad frames you just have to return list(zip(distances, nr_of_bad_frames)) - Everything should stay the same
        return list(zip(work, similarities))

    def evaluate_ind(individual):
        return 0.0, 0.0

    def save_and_show(generation):
        # Saving the final results:
        final_df = pd.DataFrame([ind for ind in hof])
        final_df.to_csv(f"./CSVs_FINAL_2/nsga2-final-pareto_{generation}.csv", index=False)

        # Getting the final results:
        pareto_fitnesses = [ind.fitness.values for ind in hof]
        objective_1_work = [fit[0] for fit in pareto_fitnesses]
        objective_2_distance = [fit[1] for fit in pareto_fitnesses]
        objective_3_similarity = [fit[2] for fit in pareto_fitnesses]
        pd.DataFrame({
            "objective_1_work": objective_1_work,
            "objective_2_distance": objective_2_distance,
            "objective_2_similarity": objective_3_similarity,
            "genotype": [ind for ind in hof]
        }).to_csv(f"./CSVs_FINAL_2/pareto_scores_{generation}.csv")
        # import pdb;pdb.set_trace()

        # Create a figure with 4 subplots
        fig = plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        plt.scatter(objective_2_distance, objective_1_work, color='blue')
        plt.ylabel('Objective 1: Work (minimize)')
        plt.xlabel('Objective 2: Distance (maximize)')
        plt.title('Work vs Distance')
        plt.ticklabel_format(style='plain')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.scatter(objective_3_similarity, objective_1_work, color='red')
        plt.ylabel('Objective 1: Work (minimize)')
        plt.xlabel('Objective 3: Similarity (minimize)')
        plt.title('Work vs Similarity')
        plt.ticklabel_format(style='plain')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.scatter(objective_3_similarity, objective_2_distance, color='green')
        plt.ylabel('Objective 2: Distance (maximize)')
        plt.xlabel('Objective 3: Similarity (minimize)')
        plt.title('Distance vs Similarity')
        plt.ticklabel_format(style='plain')
        plt.grid(True)
        
        ax = plt.subplot(2, 2, 4, projection='3d')
        ax.scatter(objective_1_work, objective_2_distance, objective_3_similarity, color='purple')
        ax.set_xlabel('Objective 1: Work (minimize)')
        ax.set_ylabel('Objective 2: Distance (maximize)')
        ax.set_zlabel('Objective 3: Similarity (minimize)')
        plt.ticklabel_format(style='plain')
        ax.set_title('3D Pareto Front')
        
        plt.tight_layout()
        plt.savefig(f"./CSVs_FINAL_2/nsga2_pareto_plot_{generation}.png")
        # plt.show(block=False)
        plt.close()
    def save_and_show_2obj(generation):
        # Saving the final results:
        final_df = pd.DataFrame([ind for ind in hof])
        final_df.to_csv(f"./CSVs_FINAL_2/nsga2-final-pareto_{generation}.csv", index=False)

        # Getting the final results:
        pareto_fitnesses = [ind.fitness.values for ind in hof]
        objective_1_work = [fit[0] for fit in pareto_fitnesses]
        objective_2_similarity = [fit[1] for fit in pareto_fitnesses]
        pd.DataFrame({
            "objective_1_work": objective_1_work,
            "objective_2_similarity": objective_2_similarity,
            "genotype": [ind for ind in hof]
        }).to_csv(f"./CSVs_FINAL_2/pareto_scores_{generation}.csv", index=False)
        
        # import pdb;pdb.set_trace()

        # Create a figure with 4 subplots
        fig = plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        plt.scatter(objective_2_similarity, objective_1_work, color='blue')
        plt.ylabel('Objective 1: Work (minimize)')
        plt.xlabel('Objective 2: Distance (maximize)')
        plt.title('Work vs Distance')
        plt.ticklabel_format(style='plain')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"./CSVs_FINAL_2/nsga2_pareto_plot_{generation}.png")
        # plt.show(block=False)
        plt.close()

    def update_means_and_stds(): # Not used
        all_distances = np.concatenate(distances)
        all_similarities = np.concatenate(symmetries)
        mean_dist = np.mean(all_distances)
        std_dist = np.std(all_distances)
        mean_sim = np.mean(all_similarities)
        std_sim = np.std(all_similarities)
# ----------------------------------------------------------------------------------------------------------------------
    # (min work, max distance, min similarity)
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -2.5, 2.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NUMBER_OF_GENES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
    #distances.append([np.array(fitnesses)[:, 0]])
    #symmetries.append([np.array(fitnesses)[:, 1]])

    # Start the learning process
    for gen in range(NGEN):
        print(f"Generation: {gen+1}/{NGEN}")

        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.2)
        for i in range(len(offspring)):
            for j in range(NUMBER_OF_GENES):
                offspring[i][j] = min(max(offspring[i][j], -2.5), 2.5)

        # We calculate only the offsprings fitness
        #update_means_and_stds()
        fitnesses = evaluate_population(offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        #distances.append([np.array(fitnesses)[:, 0]])
        #symmetries.append([np.array(fitnesses)[:, 1]])

        pop = toolbox.select(offspring + pop, k=POP_SIZE)
        hof.update(pop)

        # We save the results
        df_scores = pd.DataFrame(fitnesses, columns=["objective_1_work", "objective_3_similarity", "objective_2_distance"])
        df_scores["objective_1_work"] = [fit[0] for fit in fitnesses]
        df_scores["objective_2_distances"] = [fit[1] for fit in fitnesses]
        df_scores["objective_3_similarity"] = [fit[2] for fit in fitnesses]
        df_scores["generation"] = gen
        df_scores["genotype"] = [json.dumps(ind) for ind in pop]
        df_scores = df_scores.sort_values(by="objective_3_similarity", ascending=True).reset_index(drop=True)

        # df_scores = pd.DataFrame(fitnesses, columns=["objective_1_work", "objective_2_similarity"])
        # df_scores["objective_1_work"] = [fit[0] for fit in fitnesses]
        # df_scores["objective_2_similarity"] = [fit[1] for fit in fitnesses]
        # df_scores["generation"] = gen
        # df_scores["genotype"] = [json.dumps(ind) for ind in pop]
        # df_scores = df_scores.sort_values(by="objective_2_similarity", ascending=True).reset_index(drop=True)

        filename = f"./CSVs_2/nsga2-gen-{gen}-pareto.csv"
        df_scores.to_csv(filename, index=False)


        # if (gen + 1) % 10 == 0:
        save_and_show(gen)

    save_and_show(NGEN)
