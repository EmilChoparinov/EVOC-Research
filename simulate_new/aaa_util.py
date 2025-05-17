import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor, BrainCpgNetworkStatic
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from simulate_new import data, stypes, evaluate
from simulate_new.cmaes_ea import simulate_solutions, simulate_simple
from simulate_new.evaluate import calculate_4_angles

def optimize_individual(genotype, std_dev=0.005):
    best_fitness = evaluate_individual_distance(genotype)
    best_genotype = genotype
    ok = False
    while not ok:
        ok = True
        for _ in range(32):
            x = best_genotype.copy()
            for i in range(9):
                x[i] += np.random.normal(loc=0, scale=std_dev)
                x[i] = min(max(x[i], -2.5), 2.5)
            fitness = evaluate_individual_distance(x)
            if fitness > best_fitness:
                best_fitness = fitness
                best_genotype = x
                ok = False
                break
    return best_genotype
# ----------------------------------------------------------------------------------------------------------------------
def visualize_individual(genotype):
    body_shape = gecko_v2()
    cpg_network_struct, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    robot = ModularRobot(body=body_shape, brain=BrainCpgNetworkStatic.uniform_from_params(
        params=genotype,
        cpg_network_structure=cpg_network_struct,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping))

    scenes = []
    s = ModularRobotScene(terrain=terrains.flat())
    s.add_robot(robot)
    scenes.append(s)

    simulate_scenes(
        simulator=LocalSimulator(headless=False, num_simulators=1, start_paused=True),
        batch_parameters=make_standard_batch_parameters(
            simulation_time=30,
            sampling_frequency=30
        ),
        scenes=scenes
    )
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_individual_distance(genotype):
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
    state = stypes.EAState(
        generation=1,
        run=1,
        alpha=-1,
        animal_data=animal_data,
    )

    robot, behavior = simulate_simple(genotype)
    df_behavior = data.behaviors_to_dataframes(robot, behavior, state, z_axis=True)
    f = -evaluate.evaluate_by_distance(df_behavior[0])
    return f

def evaluate_individual_similarity_4(genotype):
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
    state = stypes.EAState(
        generation=1,
        run=1,
        alpha=-1,
        animal_data=animal_data
    )

    robot, behavior = simulate_simple(genotype)
    df_behavior = data.behaviors_to_dataframes(robot, behavior, state, z_axis=True)

    f = evaluate.evaluate_by_4_angles(df_behavior[0], state.animal_data)
    return f
# ----------------------------------------------------------------------------------------------------------------------

def plot_4_angles_comparison(robot_angles, animal_angles):
    num_angles = 4
    time = range(901)

    for i in range(num_angles):
        plt.figure()
        plt.plot(time, robot_angles[i], label='Robot', color='blue')
        plt.plot(time, animal_angles[i], label='Animal', color='orange')
        plt.title(f'Angle {i+1} Comparison Over Time')
        plt.xlabel('Time (frames)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def compare_with_animal_4(genotype):
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
    state = stypes.EAState(
        generation=0,
        run=1,
        alpha=-1,
        animal_data=animal_data[:901],
    )

    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    config = stypes.EAConfig(ttl=30, freq=30)

    robot, behavior = simulate_solutions([genotype], cpg_struct, body_shape, mapping, config)
    df = data.behaviors_to_dataframes(robot, behavior, state, z_axis=True)[0]

    robot_angles = list(zip(*calculate_4_angles(df)))
    animal_angles = list(zip(*calculate_4_angles(state.animal_data)))

    plot_4_angles_comparison(robot_angles, animal_angles)
# ----------------------------------------------------------------------------------------------------------------------

animal_data_file = "Files/slow_lerp_2.csv"
genotype = \
[0.03546062464240426, -1.5047713865129424, 0.10836346871518059, 0.6932487611896359, 0.3655277043580059, 0.19805567670626756, 2.373290847688154, -0.06133251145166401, -1.2031692287980407]


print(evaluate_individual_distance(genotype), evaluate_individual_similarity_4(genotype))
visualize_individual(genotype)
#compare_with_animal_4(genotype)

# Every single cpg has a different scaling factor -> The distance is not objective
# Scaling factor doesn't modify animal similarity
