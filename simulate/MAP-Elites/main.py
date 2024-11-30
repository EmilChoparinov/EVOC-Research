import json

import config
from util_fitness_and_metrics import calculate_fitness_and_metrics
from util_operations import select_survivors, select_parents, crossover, mutate_and_calculate_fitness_and_metrics
from util import generate_initial_population, construct_robots_and_simulate_behaviors


def save_best_individual(file_path, best_individual_fitness, best_individual, best_individual_metrics):
    with open(file_path, "w") as f:
        f.write(f"Best individual fitness = {best_individual_fitness}\n")
        f.write(f"Best individual = {best_individual}\n")
        f.write(f"Metrics: {best_individual_metrics}\n")
    print(f"Upgraded individual saved to {file_path}.")

def train():
    print("The initialization takes a while...")
    population = generate_initial_population(population_size=config.max_population_size)
    robots, behaviors = construct_robots_and_simulate_behaviors(population)
    population_fitness_and_metrics = calculate_fitness_and_metrics(robots, behaviors)

    with open("Individuals_[-10, 10]/best_fitness.json", "r") as f:
        best_fitness = json.load(f)
    population, population_fitness_and_metrics, buckets = select_survivors(population, population_fitness_and_metrics)
    for generation in range(config.nr_of_generations):
        # Print stats:
        print("-----------------------------------------------------------")
        print(f"Generation {generation + 1} / {config.nr_of_generations}.")
        for i in range(len(population)):
            bucket_string = str(buckets[i])
            if best_fitness.get(bucket_string, 0) < population_fitness_and_metrics[i][0]:
                best_fitness[bucket_string] = population_fitness_and_metrics[i][0]
                save_best_individual(f"Individuals_[-10, 10]/best_individual_{int(buckets[i])}.txt",
                                     best_fitness[bucket_string], population[i], population_fitness_and_metrics[i][1:])
                with open("Individuals_[-10, 10]/best_fitness.json", "w") as file:
                    json.dump(best_fitness, file, indent=4)

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

