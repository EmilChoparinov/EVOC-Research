import json

import config
from util_fitness_and_metrics import calculate_fitness_and_metrics
from util_operations import select_survivors, select_parents, crossover, mutate_and_calculate_fitness_and_metrics
from util import generate_initial_population, construct_robots_and_simulate_behaviors


def save_best_individual(file_path, best_individual_fitness, best_individual, best_individual_metrics):
    with open(file_path, "w") as f:
        f.write(f"Best individual fitness = {best_individual_fitness:.5f}\n")
        f.write(f"Best individual = {best_individual}\n")
        f.write(f"Metrics: {best_individual_metrics}\n")
    print(f"Best individual saved to {file_path}.")

def train():
    population = generate_initial_population(population_size=config.population_size)
    robots, behaviors = construct_robots_and_simulate_behaviors(population)
    population_fitness_and_metrics = calculate_fitness_and_metrics(robots, behaviors)

    with open("Individuals/best_fitness.json", "r") as f:
        best_fitness = json.load(f)
    population, population_fitness_and_metrics, buckets = select_survivors(population, population_fitness_and_metrics)
    for generation in range(config.nr_of_generations):
        # Print stats:
        print("-----------------------------------------------------------")
        print(f"Generation {generation + 1} / {config.nr_of_generations}.")
        for i in range(len(population)):
            if best_fitness.get(buckets[i], 0) < population_fitness_and_metrics[i][0]:
                best_fitness[buckets[i]] = population_fitness_and_metrics[i][0]
                save_best_individual(f"Individuals/best_individual_{int(buckets[i])}.txt",
                                     best_fitness[buckets[i]], population[i], population_fitness_and_metrics[i][1:])
                with open("Individuals/best_fitness.json", "w") as f:
                    json.dump(best_fitness, f)

        fitness_values = [population_fitness_and_metrics[i][0] for i in range(len(population))]
        best_individual_fitness = max(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        print(f"Population size = {len(population)}")
        print(f"Best fitness = {best_individual_fitness}")
        print(f"Average fitness = {average_fitness}")
        #----------------------------------------------------------------------------------------------------------------------------

        parents, parents_fitness_and_metrics = select_parents(population, population_fitness_and_metrics)
        offsprings = crossover(parents)
        offsprings, offsprings_fitness_and_metrics = mutate_and_calculate_fitness_and_metrics(offsprings)

        total_population = population + offsprings
        total_fitness_and_metrics = population_fitness_and_metrics + offsprings_fitness_and_metrics
        population, population_fitness_and_metrics, buckets = select_survivors(total_population, total_fitness_and_metrics)


train()

