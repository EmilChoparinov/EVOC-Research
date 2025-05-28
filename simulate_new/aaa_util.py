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
from simulate_new.evaluate import calculate_1_angle_single, calculate_2_angles, calculate_4_angles


def evaluate_individual(genotype, objective):
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))[:901]
    state = stypes.EAState(
        generation=1,
        run=1,
        alpha=-1,
        animal_data=animal_data,
    )

    robot, behavior = simulate_simple(genotype)
    df_behavior = data.behaviors_to_dataframes(robot, behavior, state, z_axis=True)
    match objective:
        case "Distance":
            return evaluate.evaluate_by_distance(df_behavior[0])
        case "1_Angle":
            return evaluate.evaluate_by_1_angle(df_behavior[0], animal_data)
        case "2_Angles":
            return  evaluate.evaluate_by_2_angles(df_behavior[0], animal_data)
        case "4_Angles":
            return evaluate.evaluate_by_4_angles(df_behavior[0], animal_data)
        case "All_Angles":
            return evaluate.evaluate_by_all_angles(df_behavior[0], animal_data)
# ----------------------------------------------------------------------------------------------------------------------
def optimize_individual(genotype, objective, std_dev=0.005):
    best_fitness = evaluate_individual(genotype, objective)
    best_genotype = genotype
    ok = False
    while not ok:
        ok = True
        for _ in range(32):
            x = best_genotype.copy()
            for i in range(9):
                x[i] += np.random.normal(loc=0, scale=std_dev)
                x[i] = min(max(x[i], -2.5), 2.5)

            fitness = evaluate_individual(x, objective)
            if fitness < best_fitness:
                best_fitness = fitness
                best_genotype = x
                ok = False
                break
    return best_genotype

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

def plot_angles_comparison(robot_angles, animal_angles):
    num_angles = len(robot_angles)
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

def compare_with_animal(genotype, objective):
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))[:901]
    state = stypes.EAState(
        generation=0,
        run=1,
        alpha=-1,
        animal_data=animal_data,
    )

    body_shape = gecko_v2()
    cpg_struct, mapping = active_hinges_to_cpg_network_structure_neighbor(
        body_shape.find_modules_of_type(ActiveHinge))

    config = stypes.EAConfig(ttl=30, freq=30)

    robot, behavior = simulate_solutions([genotype], cpg_struct, body_shape, mapping, config)
    df = data.behaviors_to_dataframes(robot, behavior, state, z_axis=True)[0]

    match objective:
        case "1_Angle":
            animal_angles = list(zip(*calculate_1_angle_single(state.animal_data)))
            robot_angles = list(zip(*calculate_1_angle_single(df)))
        case "2_Angles":
            animal_angles = list(zip(*calculate_2_angles(state.animal_data)))
            robot_angles = list(zip(*calculate_2_angles(df)))
        case "4_Angles":
            animal_angles = list(zip(*calculate_4_angles(state.animal_data)))
            robot_angles = list(zip(*calculate_4_angles(df)))

    plot_angles_comparison(robot_angles, animal_angles)
# ----------------------------------------------------------------------------------------------------------------------

"""
[-2.2702701795814058, 1.4197192191969803, -0.2771003797079275, 0.19956715517513804, -1.1571617291228409, 
-0.6970421300149127, -2.4997239807421288, 1.6553589246039713, 2.068244839256366]
The best genotype for ./Files/animal_data_3_slow_down_lerp_3.csv
7.926095782809224

[-1.0661599122893222, 0.1351742294213528, -0.5326182029142613, 0.1714331837442551, 0.021988344900795247,
-1.1641712106318418, 0.7454864014823631, 2.156757113536335, -1.6450937751867878]
The best genotype for ./Files/animal_data_3_slow_down_lerp_2.csv
9.011201890831028
"""
animal_data_file = "./Files/animal_data_3_slow_down_lerp_3.csv"
genotype = []

distance = None
One_Angle = None
Two_Angles = None
Four_Angels = None

#distance = evaluate_individual(genotype, "Distance")
One_Angle = evaluate_individual(genotype, "1_Angle")
#Two_Angles = evaluate_individual(genotype, "2_Angles")
#Four_Angels = evaluate_individual(genotype, "4_Angles")
print(f"Distance: {distance}, 1_Angle: {One_Angle}, 2_Angles: {Two_Angles}, 4_Angles: {Four_Angels}")

compare_with_animal(genotype, "1_Angle")
visualize_individual(genotype)
