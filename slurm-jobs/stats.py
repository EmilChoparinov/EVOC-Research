import os
import pandas as pd
from scipy.stats import shapiro, levene, f_oneway, kruskal
import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering plots
import seaborn as sns


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
    Load all nested 'results-*' folders in the base directory and extract the best rows based on 'Distance' from each CSV,
    while ensuring the alphas are sorted in the correct numeric order.
    """
    data_by_folder = {}
    for root, dirs, files in os.walk(base_dir):
        # Check for nested "results-*" folders in all levels
        if "results-" in root and files:
            print(f"Processing folder: {root}")  # Debugging print
            best_rows = []
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    print(f"Reading file: {file_path}")  # Debugging print
                    try:
                        df = pd.read_csv(file_path)
                        if 'Distance' in df.columns:  # Ensure the 'Distance' column exists
                            best_row = df.loc[df['Distance'].idxmax()]  # Find the row with the highest Distance
                            best_rows.append(best_row)
                        else:
                            print(f"Skipping file {file_path}, no 'Distance' column.")  # Debugging print
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")  # Debugging print
            if best_rows:
                folder_key = os.path.relpath(root, base_dir)  # Use relative path as folder name
                data_by_folder[folder_key] = pd.DataFrame(best_rows)

    # Sort the folders by their numeric 'alpha' value extracted from the folder name
    sorted_data_by_folder = {}
    for key in sorted(data_by_folder.keys(), key=lambda x: float(x.split('-')[1])):
        sorted_data_by_folder[key] = data_by_folder[key]

    return sorted_data_by_folder

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


from scipy.stats import linregress

def combined_scatter_with_densities_and_fit_results(data_by_folder):
    """
    Create a single combined scatter plot with marginal density plots
    for all folders, color-coded by folder, display regression results below the plot,
    and clip axes to the range [0, 1].
    """
    # Set up the figure
    fig = plt.figure(figsize=(10, 12))  # Extra height for fit results
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)  # Grid for scatter + marginal plots

    # Axes for scatter plot
    scatter_ax = fig.add_subplot(grid[1:4, 0:3])

    # Axes for marginal densities
    x_density_ax = fig.add_subplot(grid[0, 0:3], sharex=scatter_ax)
    y_density_ax = fig.add_subplot(grid[1:4, 3], sharey=scatter_ax)

    # Colors for the folders
    hard_coded_colors = ['red', 'blue', 'green', 'orange', 'purple']
    num_folders = len(data_by_folder)

    if num_folders > len(hard_coded_colors):
        logging.warning("More folders than available hard-coded colors. Some folders may share the same color.")

    # Extract numeric part of folder names for the legend
    legend_labels = [folder.split('-')[1] for folder in data_by_folder.keys()]

    # Combined data for regression
    combined_distances = []
    combined_similarities = []

    for color, folder, label in zip(hard_coded_colors[:num_folders], data_by_folder.keys(), legend_labels):
        df = data_by_folder[folder]

        # Append data for regression
        combined_distances.extend(df['Distance'])
        combined_similarities.extend(df['Animal Similarity'])

        # Scatter plot
        scatter_ax.scatter(
            df['Distance'],
            df['Animal Similarity'],
            label=label,  # Use the numeric part as label
            color=color,
            alpha=0.8,
            s=60
        )

        # X-axis density (Distance)
        sns.kdeplot(
            df['Distance'],
            color=color,
            ax=x_density_ax,
            linewidth=2,
            alpha=0.7
        )

        # Y-axis density (Animal Similarity)
        sns.kdeplot(
            df['Animal Similarity'],
            color=color,
            ax=y_density_ax,
            linewidth=2,
            alpha=0.7,
            vertical=True
        )

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(combined_distances, combined_similarities)
    line_x = np.linspace(max(0, min(combined_distances)), min(1, max(combined_distances)), 100)
    line_y = slope * line_x + intercept

    # Plot the regression line
    scatter_ax.plot(
        line_x, line_y, color='black', linestyle='--', linewidth=2, label="Regression Line"
    )

    # Clip the axes to [0, 1]
    scatter_ax.set_xlim(0, 1)
    scatter_ax.set_ylim(0, 1)

    # Customize scatter plot
    scatter_ax.set_xlabel("Distance")
    scatter_ax.set_ylabel("Animal Similarity (DTW)")
    scatter_ax.grid(True, linestyle='--', alpha=0.7)

    # Remove x ticks for the top density plot
    x_density_ax.set_yticks([])
    x_density_ax.set_ylabel("Density")

    # Remove y ticks for the side density plot
    y_density_ax.set_xticks([])
    y_density_ax.set_xlabel("Density")

    # Add legend for folders only
    scatter_ax.legend(
        title="Alpha levels",
        loc='upper right',
        bbox_to_anchor=(1.3, 1.3),  # Top-right corner
        fontsize='small'
    )

    # Add regression results below the plot
    fig.text(
        0.1, -0.02,  # X and Y positions below the plot
        f"Linear Regression Fit: $R^2$ = {r_value**2:.2f}, p-value = {p_value:.3e}, slope = {slope:.2f}, intercept = {intercept:.2f}",
        fontsize=10,
        ha='left'
    )
    fig.suptitle("Distance and Animal Similarity (DTW) for Different Alpha Levels", fontsize=14, y=1.02)

    # Save the plot
    plot_path = os.path.join(results_dir, "Combined_Scatter_with_Densities_and_Fit_Results_Clipped.pdf")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close()
    logging.info(f"Saved combined scatter plot with densities and clipped axes as PDF to {plot_path}")





# Add the new function call in the main script
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

        # Step 5: Plot scatter plot with folders as different colors
        logging.info("\nGenerating scatter plot colored by folder...")
        combined_scatter_with_densities_and_fit_results(data_by_folder)

        print(f"Results are in \"{args.base_dir}/results/\" folder :)")
