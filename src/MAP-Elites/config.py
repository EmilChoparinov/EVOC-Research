from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.modular_robots_v2 import gecko_v2


nr_of_generations = 1000000
max_population_size = 500 # Actually is 1000, but I want to make the initialization process faster.
nr_of_offsprings = 20
tournament_size_for_selection_of_the_parents = 3

body_shape = gecko_v2()

# Generate a CPG based on the gecko_v2's hinge tree
cpg_network_struct, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
    body_shape.find_modules_of_type(ActiveHinge)
)

terrain = terrains.flat()

simulator = LocalSimulator(headless=True, num_simulators=8)
simulation_ttl = 30  # In seconds
collection_rate = 30 # Number of frames per second