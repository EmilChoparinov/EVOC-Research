import os
import pandas as pd
from scipy.stats import shapiro, levene, f_oneway, kruskal
import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering plots

# Global results directory
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Configure logging for clean output
log_file_path = os.path.join(results_dir, 'results_output.txt')
logging.basicConfig(filename=log_file_path,
                    level=logging.INFO,
                    format='%(message)s')

# Step 1: Load data and extract best rows
def load_and_find_best(base_dir):
    """
    Load all nested 'results-*' folders in the base directory and extract the best rows based on 'Distance' from each CSV.
    """
    data_by_folder = {}
    for root, dirs, files in os.walk(base_dir):
        # Identify the "results-*" folders specifically
        folder_name = os.path.basename(root)
        if folder_name.startswith("results-") and files:  # Only process "results-*" folders with files
            best_rows = []
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        best_row = df.loc[df['Distance'].idxmax()]  # Find the row with the highest Distance
                        best_rows.append(best_row)
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")
            if best_rows:
                data_by_folder[folder_name] = pd.DataFrame(best_rows)
    return data_by_folder


# Step 2: Perform statistical tests
def compare_conditions(dataframes, metric):
    """
    Perform statistical tests for multiple groups.

    Parameters:
    - dataframes: A dictionary where keys are group names and values are DataFrames.
    - metric: The column name of the metric to test.
    """
    group_names = list(dataframes.keys())
    groups = [df[metric].dropna() for df in dataframes.values()]

    # Normality Tests
    shapiro_results = {name: shapiro(group).pvalue for name, group in zip(group_names, groups)}
    logging.info(f"\n*** {metric.upper()} Normality Test Results (Shapiro-Wilk) ***")
    for name, pvalue in shapiro_results.items():
        logging.info(f"Group {name}: p-value = {pvalue:.4f}")

    # Variance Homogeneity Test
    levene_pvalue = levene(*groups).pvalue
    logging.info(f"\n*** Levene's Test for Equality of Variances ***")
    logging.info(f"p-value = {levene_pvalue:.4f}")

    # Statistical Tests
    if all(p > 0.05 for p in shapiro_results.values()):
        # All groups are normally distributed
        f_stat, anova_pvalue = f_oneway(*groups)
        logging.info(f"\n*** ANOVA Results for {metric.upper()} ***")
        logging.info(f"F-statistic = {f_stat:.4f}, p-value = {anova_pvalue:.4f}")
    else:
        # Use Kruskal-Wallis test as a non-parametric alternative
        h_stat, kruskal_pvalue = kruskal(*groups)
        logging.info(f"\n*** Kruskal-Wallis Test Results for {metric.upper()} ***")
        logging.info(f"H-statistic = {h_stat:.4f}, p-value = {kruskal_pvalue:.4f}")


# Step 3: Visualization
def plot_comparison(dataframes, metric):
    """
    Plot the distributions of a given metric for all groups and save as PDF.

    Parameters:
    - dataframes: A dictionary where keys are group names and values are DataFrames.
    - metric: The column name of the metric to visualize.
    """
    group_names = list(dataframes.keys())
    groups = [df[metric].dropna() for df in dataframes.values()]

    plt.figure(figsize=(10, 6))
    plt.boxplot(groups, tick_labels=group_names)  # Updated parameter name
    plt.title(f"Comparison of {metric} Across Folders")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plot_path = os.path.join(results_dir, f"{metric}_comparison.pdf")  # Save in results directory
    plt.savefig(plot_path, format='pdf')  # Specify format
    plt.close()
    logging.info(f"Saved {metric} comparison plot as PDF to {plot_path}")


def plot_distance_similarity(dataframes):
    """
    Scatter plot of Distance vs Animal Similarity for each folder,
    with error bars representing standard deviation, and save as PDF.
    """
    means = []
    stds = []
    folder_names = []

    # Calculate means and standard deviations for each folder
    for folder, df in dataframes.items():
        mean_distance = df['Distance'].mean()
        mean_similarity = df['Animal Similarity'].mean()
        std_distance = df['Distance'].std()
        std_similarity = df['Animal Similarity'].std()

        means.append((mean_distance, mean_similarity))
        stds.append((std_distance, std_similarity))
        folder_names.append(folder)

    # Convert to arrays for easier plotting
    means = np.array(means)
    stds = np.array(stds)

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        means[:, 0], means[:, 1],
        xerr=stds[:, 0], yerr=stds[:, 1],
        fmt='o', ecolor='gray', capsize=5, label='Folders'
    )
    for i, folder in enumerate(folder_names):
        plt.text(means[i, 0], means[i, 1], folder, fontsize=10, ha='right')

    plt.title("Distance vs Animal Similarity Across Folders")
    plt.xlabel("Mean Distance")
    plt.ylabel("Mean Animal Similarity")
    plt.grid(True)
    plot_path = os.path.join(results_dir, "Distance_vs_Animal_Similarity.pdf")  # Save in results directory
    plt.savefig(plot_path, format='pdf')  # Specify format
    plt.close()
    logging.info(f"Saved Distance vs Animal Similarity plot as PDF to {plot_path}")


# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results from nested folders.")
    parser.add_argument("base_dir", type=str, help="Base directory containing 'results-*' folders")
    args = parser.parse_args()

    # Step 1: Load data and find the best rows
    logging.info("*** Loading and processing all folders ***")
    data_by_folder = load_and_find_best(args.base_dir)

    # Ensure there are at least two folders to compare
    if len(data_by_folder) < 2:
        logging.info("Not enough folders with valid data for comparison.")
    else:
        print("Processing...")
        # Step 2: Compare conditions
        logging.info("\nAnalyzing Distance across all folders...")
        compare_conditions(data_by_folder, metric='Distance')

        logging.info("\nAnalyzing Animal Similarity across all folders...")
        compare_conditions(data_by_folder, metric='Animal Similarity')

        # Step 3: Visualize results
        logging.info("\nGenerating Distance comparison plot...")
        plot_comparison(data_by_folder, metric='Distance')

        logging.info("\nGenerating Animal Similarity comparison plot...")
        plot_comparison(data_by_folder, metric='Animal Similarity')

        # Step 4: Plot Distance vs Similarity with Variance
        logging.info("\nGenerating Distance vs Animal Similarity scatter plot...")
        plot_distance_similarity(data_by_folder)


        print(f"Results are in \"{args.base_dir}/results/\" folder :)")
