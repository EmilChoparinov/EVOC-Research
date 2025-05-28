import pandas as pd

import simulate_new.stypes as stypes
import simulate_new.cmaes_ea as ea
from simulate_new import data
from revolve2.experimentation.logging import setup_logging

if __name__ == '__main__':
    generations = 500
    # Be careful to change the file also in evaluate_fast.py !!!
    animal_data_file = ea.local_path("animal_data_3_slow_down_lerp_2.csv", module="Files")
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))[:901]
    objective_type: stypes.objective_type = "1_Angle"

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