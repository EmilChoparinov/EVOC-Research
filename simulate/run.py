import ea
import config
import argparse
import glob
import os
import logging
import multiprocessing # import to run in Mac

from typing import get_args

multiprocessing.set_start_method('fork', force=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cleanup", action="store_true", 
    help="Delete *.csv, *.txt in current directory")

parser.add_argument(
    "--skip", action="store_true", 
    help="Do not perform ea iterations")

parser.add_argument(
    "--runs", type=int, default=config.ea_runs_cnt, 
    help="Times to run EA")

parser.add_argument(
    "--gens", type=int, default=config.ea_generations_cnt, 
    help="How many generations per run")

parser.add_argument(
    "--alpha", type=float, default=config.alpha, 
    help="Alpha value between 0 and 1.")

parser.add_argument(
    "--with-fit", type=str, default=config.use_fit, 
    choices=list(get_args(config.fitness_function)), 
    help="Specify the fitness function for EA.")


args = parser.parse_args()

if not (0 <= args.alpha <= 1):
    raise ValueError("Alpha value must be between 0 and 1.")

from revolve2.experimentation.logging import setup_logging
setup_logging(file_name="log.txt")

if args.cleanup:
    logging.info("Deleting *.csv, *.txt")
    for file_path in glob.glob("*.csv") + glob.glob("*.txt"): os.remove(file_path)

def main():
    # Pass the arguments to ea.process_ea_iteration
    ea.export_ea_metadata()
    ea.process_ea_iteration(
        max_gen=args.gens,
        max_runs=args.runs
    )

if __name__ == '__main__':
    if not args.skip:
        main()
