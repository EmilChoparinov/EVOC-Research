import pandas as pd

import simulate_new.stypes as stypes
import simulate_new.ea as ea
from simulate_new import data
from revolve2.experimentation.logging import setup_logging, logging

if __name__ == '__main__':
    generations = 500
    animal_data_file = ea.local_path("slow_lerp_2.csv", module="Files")
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
    objective_type: stypes.objective_type = "Distance"

    for run in range(1, 2):
        state = stypes.EAState(
            generation=generations,
            run=run,
            alpha=-1, # We don't need alpha
            animal_data=animal_data,
        )
        setup_logging(
            file_name=ea.local_path(f"{ea.file_idempotent(state, objective_type)}.txt")
        )
        config = stypes.EAConfig(ttl=30, freq=30)

        ea.optimize(state=state, config=config, objective=objective_type)