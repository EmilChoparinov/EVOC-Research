import numpy as np

from util import visualize_individual
import json
import re
import matplotlib.pyplot as plt


def extract_genotype(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    genotype = []
    capturing = False

    for line in lines:
        if line.startswith("Best individual ="):
            capturing = True
            # Start capturing genotype from this line
            match = re.search(r"\[([^\]]*)", line)
            if match:
                genotype.extend([float(num) for num in match.group(1).split()])

        elif capturing:
            # Continue capturing genotype until we reach the closing bracket
            match = re.search(r"([^\]]*)\]", line)
            if match:
                genotype.extend([float(num) for num in match.group(1).split()])
                break  # Stop once we capture the full genotype
            else:
                genotype.extend([float(num) for num in line.split()])

    return genotype

#"""
with open("Individuals/best_fitness.json", "r") as f:
    best_fitness = json.load(f)
    best_key = None
    max_fitness = 0
    genotypes = []
    for key in best_fitness.keys():
        if best_fitness[key] > 1.5:
            genotypes.append((extract_genotype(f"Individuals/best_individual_{int(float(key))}.txt"), best_fitness[key], key))
        if best_fitness[key] > max_fitness:
            max_fitness = best_fitness[key]
            best_key = key

    print(f"Number of keys: {len(best_fitness.keys())}")
    print(f"Best Key: {best_key}, Max fitness: {max_fitness}")
#"""

def plot_some_statistics():
    x_values = []
    y_values = []
    z_values = []
    t_values = []

    # Extract x, y, z, t values from the `genotypes` list
    for genotype, fitness, key in genotypes:
        key_int = int(float(key))  # Convert the key to an integer
        x = key_int // 1000
        y = (key_int // 100) % 10
        z = (key_int // 10) % 10
        t = key_int % 10
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        t_values.append(t)

    # Calculate the maximum values for matrix dimensions
    max_x = max(x_values) + 1
    max_y = max(y_values) + 1
    max_z = max(z_values) + 1
    max_t = max(t_values) + 1

    # Initialize matrices for (x, y) and (z, t) pair counts
    count_matrix_xy = np.zeros((max_x, max_y), dtype=int)
    count_matrix_zt = np.zeros((max_z, max_t), dtype=int)

    # Populate the matrices
    for x, y in zip(x_values, y_values):
        count_matrix_xy[x, y] += 1
    for z, t in zip(z_values, t_values):
        count_matrix_zt[z, t] += 1

    # Plotting the (x, y) pair count matrix
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(count_matrix_xy, cmap="Blues", origin="lower", interpolation="nearest")
    plt.colorbar(label="Count of (x, y) pairs")
    plt.title("Matrix of (x, y) Pair Counts")
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.xticks(range(max_y))
    plt.yticks(range(max_x))

    # Add text annotations for each square in (x, y) matrix
    for i in range(max_x):
        for j in range(max_y):
            if count_matrix_xy[i, j] > 0:
                plt.text(j, i, str(count_matrix_xy[i, j]), ha='center', va='center', color="black")

    # Plotting the (z, t) pair count matrix
    plt.subplot(2, 1, 2)
    plt.imshow(count_matrix_zt, cmap="Oranges", origin="lower", interpolation="nearest")
    plt.colorbar(label="Count of (z, t) pairs")
    plt.title("Matrix of (z, t) Pair Counts")
    plt.xlabel("Average movement of the limbs")
    plt.ylabel("Cycle length")
    plt.xticks(range(max_t))
    plt.yticks(range(max_z))

    # Add text annotations for each square in (z, t) matrix
    for i in range(max_z):
        for j in range(max_t):
            if count_matrix_zt[i, j] > 0:
                plt.text(j, i, str(count_matrix_zt[i, j]), ha='center', va='center', color="black")

    plt.tight_layout()
    plt.show()

plot_some_statistics()

for i in range(len(genotypes)):
    print(genotypes[i])
    if genotypes[i][1] > 1.83:
        visualize_individual(genotypes[i][0])
#visualize_individual(genotypes[1][0])