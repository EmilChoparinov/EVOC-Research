import csv
import math
import numpy as np

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
import config
from util_fitness_and_metrics import get_fitness_and_positions


def generate_initial_population(population_size):
    mean = 0.0; std_dev = 0.5
    lower_bound = -2.0; upper_bound = 2.0
    shape = (config.cpg_network_struct.num_connections,)

    population = [
        np.clip(np.random.normal(loc=mean, scale=std_dev, size=shape), lower_bound, upper_bound)
        for _ in range(population_size)
    ]
    return population

def construct_robots_and_simulate_behaviors(population):
    robots = [
        ModularRobot(
            body=config.body_shape,
            brain=BrainCpgNetworkStatic.uniform_from_params(
                params=solution,
                cpg_network_structure=config.cpg_network_struct,
                initial_state_uniform=math.sqrt(2) * 0.5,
                output_mapping=config.output_mapping
            )
        )
        for solution in population
    ]

    scenes = []
    for robot in robots:
        s = ModularRobotScene(terrain=config.terrain)
        s.add_robot(robot)
        scenes.append(s)

    return (robots, simulate_scenes(
        simulator=config.simulator,
        batch_parameters=make_standard_batch_parameters(
            simulation_time=config.simulation_ttl,
            sampling_frequency=config.collection_rate
        ),
        scenes=scenes
    ))

def visualize_individual(genotype):
    robot = ModularRobot(body=config.body_shape, brain=BrainCpgNetworkStatic.uniform_from_params(
        params=genotype,
        cpg_network_structure=config.cpg_network_struct,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=config.output_mapping))

    scenes = []
    s = ModularRobotScene(terrain=config.terrain)
    s.add_robot(robot)
    scenes.append(s)

    simulate_scenes(
        simulator=LocalSimulator(headless=False, num_simulators=1),
        batch_parameters=make_standard_batch_parameters(
            simulation_time=config.simulation_ttl,
            sampling_frequency=config.collection_rate
        ),
        scenes=scenes
    )

def get_csv_from_individual(genotype, file_path):
    individual = [genotype]
    robot, behavior = construct_robots_and_simulate_behaviors(individual)
    (_, head_positions, left_front_positions, right_front_positions, middle_positions,
     rear_positions, left_hind_positions, right_hind_positions) = get_fitness_and_positions(robot[0], behavior[0])
    frame_id = [i for i in range(len(head_positions))]

    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow([
            "frame_ID", "head", "left_front", "right_front",
            "middle", "rear", "left_hind", "right_hind"
        ])

        for i in range(len(frame_id)):
            writer.writerow([
                frame_id[i], head_positions[i], left_front_positions[i], right_front_positions[i],
                middle_positions[i], rear_positions[i], left_hind_positions[i], right_hind_positions[i]
            ])
    print(f"{file_path} saved !")
