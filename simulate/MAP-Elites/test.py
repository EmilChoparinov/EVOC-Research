from util import visualize_individual, get_csv_from_individual
import json
import re


def extract_genotype(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    genotype = []
    capturing = False
    for line in lines:
        if line.startswith("Best individual ="):
            capturing = True
            match = re.search(r"\[([^\]]*)", line)
            if match:
                genotype.extend([float(num) for num in match.group(1).split()])

        elif capturing:
            match = re.search(r"([^\]]*)\]", line)
            if match:
                genotype.extend([float(num) for num in match.group(1).split()])
                break
            else:
                genotype.extend([float(num) for num in line.split()])

    return genotype

file_path = "Individuals_[-10, 10]"
#"""
with open(f"{file_path}/best_fitness.json", "r") as f:
    best_fitness = json.load(f)
best_key = None
max_fitness = 0
genotypes = []
for key in best_fitness.keys():
    if best_fitness[key] > 3:
        genotypes.append((extract_genotype(f"{file_path}/best_individual_{int(key)}.txt"), best_fitness[key], key))
    if best_fitness[key] > max_fitness:
        max_fitness = best_fitness[key]
        best_key = key

print(f"Number of keys: {len(best_fitness.keys())}")
print(f"Best Key: {best_key}, Max fitness: {max_fitness}")
#"""

genotype = extract_genotype(f"{file_path}/best_individual_{int(best_key)}.txt")
#get_csv_from_individual(genotype, "z_example.csv")
#visualize_individual(genotype)


print(f"Number of good behaviors: {len(genotypes)}")
for i in range(len(genotypes)):
    print(genotypes[i])
    #if genotypes[i][1] == 1.7506601479825343:
    #    visualize_individual(genotypes[i][0])

#visualize_individual(genotypes[9][0])
