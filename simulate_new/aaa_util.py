import pandas as pd

from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
from revolve2.standards.modular_robots_v2 import gecko_v2
from simulate_new import data, stypes
from simulate_new.ea import simulate_solutions


def compare_with_animal(genotype):
    animal_data_file = "Files/slow_with_linear_4.csv"
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
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