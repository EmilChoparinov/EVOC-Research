import argparse
import pandas as pd
import typing
import simulate.stypes as stypes 
import simulate.ea as ea
import simulate.data as data

from revolve2.experimentation.logging import setup_logging, logging

parser = argparse.ArgumentParser()
process_pd = lambda csv: data.convert_tuple_columns(pd.read_csv(csv))
parser.add_argument(
    "--animal-data", type=process_pd, 
    help=".csv file containing animal data", required=True)

parser.add_argument(
    "--alpha", type=float, 
    required=True, help="Alpha value between 0 and 1.")

parser.add_argument(
    "--similarity-type", type=str, required=True,choices=list(stypes.similarity_type.__args__), 
    help="Specify the fitness function for EA.")

parser.add_argument(
    "--runs", type=int, 
    default=5, help="Times to run EA")

parser.add_argument(
    "--ttl", type=int,
    default=30, 
    help="The time to live (TTL) in seconds of the simulation. How long the \
        simulation will go for")

parser.add_argument("--gens", type=int, default=300, help="How many generations per run")

class Args(typing.NamedTuple):
    animal_data: pd.DataFrame
    alpha: float
    similarity_type: stypes.similarity_type
    gens: int
    runs: int
    ttl: int

args: Args = parser.parse_args()

if not (0 <= args.alpha <= 1): raise ValueError("Alpha must be bounded [0,1]")

if __name__ == '__main__':
    state = ea.create_state(
            generation=args.gens, run=args.runs,
            alpha=args.alpha, 
            animal_data=args.animal_data, similarity_type=args.similarity_type)
    
    setup_logging(file_name=f"{ea.file_idempotent(state)}.txt")
    ea.iterate(state,
               ea.create_config(ttl=30, freq=30))