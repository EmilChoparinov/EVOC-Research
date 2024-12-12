import cv2
from collections import namedtuple
from functools import reduce
import itertools
import operator
import math
import os
import json
import argparse
import numpy as np
from typing import NamedTuple
#=== Types
class Sample(NamedTuple):
    img: cv2.UMat
    label: any
    id: str

class Args(NamedTuple):
    indir : str
    outdir: str
    ext   : str
    group : int

#=== Collect Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "indir", type=str, help="Folder containing a labeled dataset")
parser.add_argument(
    "--outdir", type=str, help="Folder to output augmented dataset", default="dataset-augmented")
parser.add_argument(
    "--group", type=int, help="How many times the image should be rotated and saved", default=3)
parser.add_argument(
    "--ext", type=str, help="The file extension of the images", default='.rgb.jpg')

args: Args = parser.parse_args()

#=== Utils
def load_sample(name: str) -> Sample:
    with open(os.path.join(args.indir, f'{name}.json')) as f: 
        label = json.load(f)
    return Sample(
        img=cv2.imread(os.path.join(args.indir, f'{name}{args.ext}')), 
        label=label,
        id=name)

def save_sample(sample: Sample) -> None:
    cv2.imwrite(os.path.join(args.outdir, f"{sample.id}{args.ext}"), sample.img)
    with open(os.path.join(args.outdir, f'{sample.id}.json'), 'w') as f:
        json.dump(sample.label, f)

#=== Program Begin
os.makedirs(args.outdir, exist_ok=True)

# Collect dataset
ids = list(set([int(filename.split('.')[0]) for filename in os.listdir(args.indir)]))
dataset = [load_sample(name) for name in ids]    

if(len(dataset) == 0): raise Exception(
    f"Error: Dataset is empty! Dataset should be populated in folder: {args.indir}")

def rotate_point_type(point_type, M: cv2.typing.MatLike):
    point_type["projected_location"][:] = [
        np.around(np.dot(M, np.array([point[0], point[1], 1]))).astype(int).tolist()
            for point in point_type["projected_location"]]

def apply_rotation(sample: Sample, rot: float, cnt: int) -> Sample:
    mut_new_label = json.loads(json.dumps(sample.label))

    h, w, *_ = sample.img.shape

    # Collect the rotation matrix and apply it to each label accordingly
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
    [rotate_point_type(point_type, M) 
        for point_type in mut_new_label["objects"][0]["keypoints"]]

    # Return a new sample that has warped its image according to do rotation
    # matrix `M`
    return Sample(
        img=cv2.warpAffine(sample.img, M, (w, h)),
        label=mut_new_label,
        id=f"{sample.id}-{cnt}")

def apply_transform(sample: Sample, times: int = 0) -> list[Sample]:
    to_rotate = [(rotate_value, times) 
                    for rotate_value, times in 
                    zip(
                        np.random.uniform(-180, 180, times), 
                        np.arange(1, times + 1))]
    
    return [apply_rotation(sample, rot_val, time) for rot_val, time in to_rotate]

augmented_dataset = \
    list(itertools.chain.from_iterable(
        [apply_transform(sample, args.group) for sample in dataset]))

# Add to augmented dataset the original dataset and output to folder
augmented_dataset += dataset
[save_sample(sample) for sample in augmented_dataset]