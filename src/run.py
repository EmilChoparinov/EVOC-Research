import ea
import config
import argparse
import glob
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--cleanup", action="store_true", help="Delete *.csv, *.txt in current directory")
parser.add_argument("--skip", action="store_true", help="Do not perform ea iterations")
parser.add_argument("--runs", type=int, default=config.ea_runs_cnt, help="Times to run EA")
parser.add_argument("--gens", type=int, default=config.ea_generations_cnt, help="How many generations per run")

args = parser.parse_args()

from revolve2.experimentation.logging import setup_logging
# Why?? I have to run this before anything else to initiate logging. Consider 
# this a NoOP. Logging is handeled independently by each iteration
setup_logging()

if(args.cleanup):
    logging.info("Deleting *.csv, *.txt")
    for file_path in glob.glob("*.csv") + glob.glob("*.txt"): os.remove(file_path)

if not args.skip:
    ea.process_ea_iteration(args.gens, args.runs)