import cv2
import os
import logging
import argparse
from revolve2.experimentation.logging import setup_logging
from functools import reduce
from typing import NamedTuple

#=== Typedef
setup_logging()
class Args(NamedTuple):
    indir:  str
    outdir: str
    ext:    str
    skip:   int

#=== Argument parser routine
def valid_dir_path(path: str) -> str:
    if os.path.exists(path) and not os.path.isdir(path):
        raise argparse.ArgumentTypeError("ERROR: path given is invalid")
    return path

parser = argparse.ArgumentParser()

parser.add_argument(
    "indir", type=valid_dir_path,
    help="Folder containing *.[mp4|mov] files" 
)

parser.add_argument(
    "--skip", type=int, default=16,
    help="take only every N frame into the dataset"
)

parser.add_argument(
    "--outdir", type=valid_dir_path, default="./dataset",
    help="Specify the output directory of the dataset"
)

parser.add_argument(
    "--ext",  type=str, default=".rgb.jpg",
    help="The file extension of the images saved"
)

args: Args = parser.parse_args()

#=== Validate videos routine

# Select the point where we last stopped in the dataset
ending_f_id = max(
    [int(file.split('.')[0]) 
        for file in os.listdir(args.indir) if file.endswith(args.ext)],
    default=0
)

if not os.path.isdir(args.indir):
    raise Exception(
        f"`{args.indir}` is an invalid directory. It must exist and contain only .[mp4|mov] files.")

os.makedirs(args.outdir, exist_ok=True)

#=== Conversion routine
def add_to_dataset(gf_id: int, video: str) -> int:
    c = cv2.VideoCapture(os.path.join(args.indir, video))
    if not c.isOpened():
        logging.error(f"Error: Cannot open file: {video}. Skipping")
        return gf_id

    logging.info(f"Initiate conversion for file: {video}")
    # Since we only care about frame data in the dataset. `f_id` iterates over
    # the frames of the video, while it gets saved on `gf_id` (global frame id).
    f_id = 0
    while c.isOpened():
        ret, frame = c.read()
        if not ret: 
            logging.info(f"Error: Could not get next frame for file: {video}. Skipping")
            return gf_id
        
        if not f_id % args.skip == 0:
            f_id += 1
            continue

        logging.info(f"Write as frame: {gf_id} (id: {f_id})")
        cv2.imwrite(f"{args.outdir}/{gf_id}{args.ext}", frame)

        # NOTE! The program assumes that the dataset being outputed is ALWAYS
        # unique. We do not try to iterate `f_id` up to `g_id`.
        gf_id += 1
        f_id += 1

    logging.info(f"Conversion complete for file: {video}")    
    c.release()
    return gf_id

# Reduce the videos into the dataset of images
reduce(add_to_dataset, [video for video in os.listdir(args.indir)], ending_f_id)

logging.info("Conversion complete! Exiting")