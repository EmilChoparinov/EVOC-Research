import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_fitness_data(file_paths, alpha):
    fitness_data_distance = {}
    fitness_data_similarity = {}
    fitness_data_best = {}

    for file_path in file_paths:
        df = pd.read_csv(file_path, usecols=[0, 1])

        # Convert columns to numeric, forcing non-numeric values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Print the data types of the columns to debug mixed types issue
        print(f"Data types for file {file_path}:")
        print(df.dtypes)

        print(df.head())
        # Check for required columns and initialize dictionaries if columns exist
        if 'generation_id' not in df.columns:
            raise ValueError(f"CSV file at {file_path} is missing 'generation_id' column.")

        has_distance_fitness = 'distance_fitness' in df.columns
        has_similarity_fitness = 'similarity_fitness' in df.columns
        has_best_fitness = 'generation_best_fitness_score' in df.columns
        print(df.columns)

        # Group data by generation and take the first row for each generation_id
        generation_groups = df.groupby('generation_id').first()
        for generation, row in generation_groups.iterrows():
            if has_distance_fitness:
                distance_fitness = row['distance_fitness']
                if generation not in fitness_data_distance:
                    fitness_data_distance[generation] = []
                fitness_data_distance[generation].append(distance_fitness)

            if has_similarity_fitness:
                similarity_fitness = row['similarity_fitness']
                if generation not in fitness_data_similarity:
                    fitness_data_similarity[generation] = []
                fitness_data_similarity[generation].append(similarity_fitness)

            if has_best_fitness:
                best_fitness = row['generation_best_fitness_score']
                if generation not in fitness_data_best:
                    fitness_data_best[generation] = []
                fitness_data_best[generation].append(best_fitness)

    return fitness_data_distance, fitness_data_similarity, fitness_data_best


def calculate_fitness_statistics(fitness_data):
    statistics = {}
    for generation, fitness_values in fitness_data.items():
        fitness_values_sorted = sorted(fitness_values)
        mean_fitness = np.mean(fitness_values_sorted)
        percentile_25 = np.percentile(fitness_values_sorted, 25)
        percentile_75 = np.percentile(fitness_values_sorted, 75)

        # Ensure the mean is within the 25th and 75th percentiles
        if not (percentile_25 <= mean_fitness <= percentile_75):
            raise ValueError(f"Mean fitness {mean_fitness} is not between the 25th percentile {percentile_25} and 75th percentile {percentile_75} for generation {generation}")

        statistics[generation] = {
            'mean': mean_fitness,
            '25th_percentile': percentile_25,
            '75th_percentile': percentile_75
        }

    return statistics
def plot_fitness_statistics(fitness_stats_by_alpha, fitness_type):
    plt.figure(figsize=(10, 6))

    for alpha, stats in fitness_stats_by_alpha.items():
        generations = sorted(stats.keys())
        means = [stats[gen]['mean'] for gen in generations]
        percentile_25 = [stats[gen]['25th_percentile'] for gen in generations]
        percentile_75 = [stats[gen]['75th_percentile'] for gen in generations]

        plt.plot(generations, means, label=f'Alpha = {alpha:.2f}')
        plt.fill_between(generations, percentile_25, percentile_75, alpha=0.2)

    plt.title(f'Fitness Evolution ({fitness_type.capitalize()})')
    plt.xlabel('Generation')
    plt.ylabel(f'{fitness_type.capitalize()} Fitness')
    plt.legend(title='Alpha Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(alpha_values, file_paths_by_alpha):
    fitness_stats_by_alpha_distance = {}
    fitness_stats_by_alpha_similarity = {}
    fitness_stats_by_alpha_best = {}

    for alpha in alpha_values:
        file_paths = file_paths_by_alpha[alpha]
        fitness_data_distance, fitness_data_similarity, fitness_data_best = load_fitness_data(file_paths, alpha)
        print(fitness_data_best)

        # Calculate statistics for distance, similarity, and best fitness
        fitness_statistics_distance = calculate_fitness_statistics(fitness_data_distance)
        fitness_statistics_similarity = calculate_fitness_statistics(fitness_data_similarity)
        fitness_statistics_best = calculate_fitness_statistics(fitness_data_best)

        print(fitness_statistics_best)
        # Store statistics by alpha for plotting
        fitness_stats_by_alpha_distance[alpha] = fitness_statistics_distance
        fitness_stats_by_alpha_similarity[alpha] = fitness_statistics_similarity
        fitness_stats_by_alpha_best[alpha] = fitness_statistics_best

    # Debug prints to verify the data being passed to the plotting functions
    print("Fitness stats by alpha (distance):", fitness_stats_by_alpha_distance)
    print("Fitness stats by alpha (similarity):", fitness_stats_by_alpha_similarity)
    print("Fitness stats by alpha (best):", fitness_stats_by_alpha_best)

    # Plot distance, similarity, and best fitness separately
    plot_fitness_statistics(fitness_stats_by_alpha_distance, 'distance')
    plot_fitness_statistics(fitness_stats_by_alpha_similarity, 'similarity')
    plot_fitness_statistics(fitness_stats_by_alpha_best, 'best')

# Define the folder path
folder_path = r"C:\Users\TJ\Desktop\2024-2025\Evolutionary Computing\Research Assignment\Revolve2\apps\plot\runs"

# Gather all CSV file paths in the directory
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Define alpha values and file paths for each alpha
alpha_values = [1]
file_paths_by_alpha = {
    1: csv_files,
}

# Call the main function
main(alpha_values, file_paths_by_alpha)