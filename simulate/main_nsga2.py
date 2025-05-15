import pandas as pd
import simulate.stypes as stypes
import simulate.ea_nsga2 as ea_nsga2
import simulate.data as data

if __name__ == '__main__':
    generations = 250
    runs = 1
    animal_data_file = "./simulate/model/slow_with_linear_4.csv"
    animal_data = data.convert_tuple_columns(pd.read_csv(animal_data_file))
    similarity_type: stypes.similarity_type = "4_Angles"

    state = stypes.EAState(
        generation=generations,
        run=runs,
        alpha=-1, # We don't need alpha
        animal_data=animal_data,
        similarity_type=similarity_type,
    )
    config = stypes.EAConfig(ttl=30, freq=30)

    ea_nsga2.nsga2_optimize(state=state, config=config)
