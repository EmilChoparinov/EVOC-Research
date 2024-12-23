import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from dataclasses import dataclass
from typing import NamedTuple
from enum import Enum

import argparse
import re

# CONSTANTS
WIDTH   = 1280
HEIGHT  = 960
FPS     = 24
SCALE_FACTOR = 300
ORIGIN  = (WIDTH // 2, HEIGHT // 2)


class RGB(NamedTuple):
    r: int
    g: int
    b: int

@dataclass(frozen=True)
class RobotTheme:
    c_box  : RGB
    c_lines: RGB

@dataclass(frozen=True)
class RobotThemes:
    sim = RobotTheme(c_box=RGB(255, 0, 0), c_lines=RGB(0,0,0))
    act = RobotTheme(c_box=RGB(0, 0, 255), c_lines=RGB(0,0,0))

@dataclass(frozen=True)
class Args:
    files : list[pd.DataFrame]
    off   : int
    name  : str
    themes: dict[int, RobotTheme]

def parse_themes(cmd: str) -> dict[int, RobotTheme]:
    regex_theme_state = re.compile(r'(\d+):(\w+),*')
    m = regex_theme_state.findall(cmd)

    if not m: 
        raise argparse.ArgumentTypeError(
            "ERROR: could not parse theme list from `--themes`")
    return {int(idx): RobotThemes.__dict__[theme_name] for idx, theme_name in m}

def parse_dataframe(path: str):
    df = pd.read_csv(path)
    df = df.loc[df['generation_id'] == 99]
    return df.reset_index(drop=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "files", type=parse_dataframe, nargs="+", help="A list of CSV files to add to the video")

parser.add_argument(
    "--off", type=int, default=200, 
    help="Offset robots equidistant by `off` pixels from each other along x")

parser.add_argument(
    "--name", type=str, default="test.mp4", help="Name of the file out")

parser.add_argument(
    "--themes", type=parse_themes, default="0:sim,1:act", 
    help="Set the themes of the robots. Currently only two are availabile: sim, and act")

# *== Globals ==*
parsed: Args = parser.parse_args()
# frame_buffer = [np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255]
frame_buffer = []
video = cv2.VideoWriter(parsed.name, cv2.VideoWriter_fourcc(*'mp4v'),\
                        FPS, (WIDTH, HEIGHT))
max_rows_to_process = min([len(file) for file in parsed.files])

def coord_to_pixel(xy: tuple[float, float], x_off: int = 0):
    return (int(ORIGIN[0] + xy[0]*SCALE_FACTOR + x_off),
            int(ORIGIN[1] - xy[1]*SCALE_FACTOR + x_off))

def apply_boxes(mat: cv2.UMat, dataset: pd.DataFrame, 
                row: int, columns: list, 
                theme: RobotTheme):
    [cv2.circle(mat, 
        coord_to_pixel(eval(dataset.at[row, col])), 
        3, theme.c_box, -1) 
    for col in columns]

def apply_edges(mat: cv2.UMat, dataset: pd.DataFrame, 
                row: int, col_pairs: list[tuple[str, str]],
                theme: RobotTheme): 
    [cv2.line(mat, 
              coord_to_pixel(eval(dataset.at[row, c1])),
              coord_to_pixel(eval(dataset.at[row, c2])),
              theme.c_lines, 1) 
              for c1, c2 in col_pairs]

def push_frames(columns: list=[]):    
    for row in range(max_rows_to_process):
        mat = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)*255
        [apply_boxes(
                mat, dataset, row, columns, parsed.themes[id]) 
            for id, dataset in enumerate(parsed.files)]
        
        [apply_edges(mat, dataset, row, 
                     [("head", "left_front"),
                      ("head", "right_front"),
                      ("head", "middle"),
                      ("rear", "left_hind"),
                      ("rear", "right_hind"),
                      ("middle", "rear")],
                     parsed.themes[id]) 
         for id, dataset in enumerate(parsed.files)]
        video.write(mat)

push_frames([
           # (face) :)
             "head",
    "left_front", "right_front", 
             "middle", 
             "rear", 
    "left_hind", "right_hind"
])

[video.write(frame) for frame in frame_buffer]
video.release()
