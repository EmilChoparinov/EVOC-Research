import pandas as pd
from matplotlib import pyplot as plt

from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
from revolve2.standards.modular_robots_v2 import gecko_v2
from simulate_new import data, stypes
from simulate_new.aaa_util2 import evaluate_individual_similarity, evaluate_individual_distance
from simulate_new.ea import simulate_solutions
from simulate_new.evaluate import calculate_4_angles

def plot_angle_comparison(robot_angles, animal_angles):
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

def compare_with_animal(genotype):
    animal_data_file = "Files/slow_lerp_4.csv"
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

genotype = [-1.36918234, -0.28729471, -2.49133755, -0.05042451, -1.6723255 ,
       -1.81931257,  2.48498   ,  1.00360881, -2.49668466]

genotype = [ 1.52459752,  2.246151  ,  1.27485924, -1.70201395,  2.23114975,
        1.47143735,  2.46921194, -0.57575046, -2.48014768]
print(evaluate_individual_similarity(genotype), evaluate_individual_distance(genotype))
compare_with_animal(genotype)
