import pandas as pd
from matplotlib import pyplot as plt

from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
from revolve2.standards.modular_robots_v2 import gecko_v2
from simulate_new import data, stypes
from simulate_new.aaa_util2 import evaluate_individual_similarity
from simulate_new.ea import simulate_solutions
from simulate_new.evaluate import calculate_4_angles

def plot_angle_comparison(robot_angles, animal_angles):
    num_angles = 4
    time = range(len(robot_angles[0]))  # assuming all have the same length
    print(time)

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

def compare_with_animal(genotype):
    animal_data_file = "Files/slow_with_linear_4.csv"
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

    plot_angle_comparison(robot_angles, animal_angles)

genotype = [-2.5, 0.8142331174729603, 0.06558948796912085,
            -2.4955815158463506, 1.6487105111015121, -2.5,
            2.495983077798169, -1.9510357390627722, -1.1433926155885599]
compare_with_animal(genotype)
print(evaluate_individual_similarity(genotype))