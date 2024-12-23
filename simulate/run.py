import ea
import VAE
import config
import argparse
import glob
import pandas as pd
import os
# import multiprocessing # import to run in Mac
from typing import NamedTuple
import typedef

from revolve2.experimentation.logging import setup_logging, logging
# multiprocessing.set_start_method('fork', force=True)

parser = argparse.ArgumentParser()
parser.add_argument("--animal-data", type=str, help=".csv file containing animal data", required=True)
parser.add_argument("--vae", type=str, help=".pth file containing the VAE", required=True)
parser.add_argument("--log", type=str, help="filename to print log out to", default="log.txt")
parser.add_argument("--cleanup", action="store_true", help="Delete *.csv, *.txt in current directory")
parser.add_argument("--skip", action="store_true", help="Do not perform ea iterations")
parser.add_argument("--runs", type=int, default=config.ea_runs_cnt, help="Times to run EA")
parser.add_argument("--gens", type=int, default=config.ea_generations_cnt, help="How many generations per run")
parser.add_argument("--alpha", type=float, default=config.alpha, help="Alpha value between 0 and 1.")
parser.add_argument("--with-fit", type=str, default=config.use_fit,choices=["distance", "similarity", "blended"], help="Specify the fitness function for EA.")
parser.add_argument("--similarity-type", type=str, default=config.type,choices=["DTW", "MSE", "Cosine","VAE","four"], help="Specify the type of similarity function.")

# no initial value, each time must ensure alpha and fitness_function
class Args(NamedTuple):
    animal_data: str
    cleanup: bool
    skip: bool
    runs: int
    gens: int
    alpha: float
    with_fit: typedef.fitness_functions
    similarity_type: typedef.similarity_type
    log: str
    vae: str

args: Args = parser.parse_args()

if not (0 <= args.alpha <= 1):
    raise ValueError("Alpha value must be between 0 and 1.")

if args.cleanup:
    print("Deleting *.csv, *.txt")
    for file_path in glob.glob("*.csv") + glob.glob("*.txt"): os.remove(file_path)

setup_logging(file_name=args.log, level=logging.INFO)

def main():
    # Pass the arguments to ea.process_ea_iteration
    ea.export_ea_metadata()
    animal_data = pd.read_csv(args.animal_data)
    ea.process_ea_iteration(
        config.create_state(
            alpha=args.alpha,
            fitness_function=args.with_fit,
            similarity_type=args.similarity_type,
            max_gen=args.gens,
            max_runs=args.runs,
            # Memoize these!! Extremely costly to do this every iteration guys!
            animal_data=animal_data,
            animal_data_infer=VAE.infer_on_csv(animal_data, args.vae)
        )
    )
    # VAE.average_and_std_plot(max_runs=args.runs,similarity_type=args.similarity_type)
if __name__ == '__main__':
    if not args.skip:
        main()
