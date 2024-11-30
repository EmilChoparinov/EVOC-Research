import cv2
import logging
import os
import json
import threading
import argparse

from typing import Callable, NamedTuple
from time import time
from dataclasses import dataclass, field
from revolve2.experimentation.logging import setup_logging
from pprint import pformat 
from random import shuffle

def valid_dir_path(path: str) -> str:
    if os.path.exists(path) and not os.path.isdir(path):
        raise argparse.ArgumentTypeError("ERROR: path given is invalid")
    return path

def get_labeling_progress(indir: str, outdir: str) -> (int, int, float):
    """Assumes that `indir` and `outdir` contain only VALID FILES"""
    total = len(os.listdir(indir))
    so_far = len(os.listdir(outdir))
    return total, so_far, so_far/total * 100 

#=== Argument Collection
parser = argparse.ArgumentParser()

parser.add_argument(
    "indir", type=valid_dir_path,
    help="Folder containing a set of images counting up in number"
)

parser.add_argument(
    "--contribute-mode", action="store_false",
    help="Contribute mode protects the current dataset stored in `markdir` and \
          outdir. It allows for you to pick up where you left off from." 
)

parser.add_argument(
    "--ordered", action="store_true",
    help="By default, the sequence of images is randomized during labeling"
)

parser.add_argument(
    "--outdir", type=valid_dir_path, default="./labeled-dataset",
    help="The output directory of the labeled dataset"
)

parser.add_argument(
    "--markdir", type=valid_dir_path, default="./marked-dataset",
    help="Contains the images with the labels burned into them" 
)

parser.add_argument(
    "--ext", type=str, default=".rgb.jpg",
    help="If the dataset contains different extensions, please change it here "\
         "or else the output will be [filename].[ext].json instead of "\
         "[filename].json"
)


#=== Program Begin
print(
"""
 _       ____  ____     ___  _      ____  ____    ____  ______   ___   ____   __ 
| |     /    ||    \   /  _]| |    |    ||    \  /    ||      | /   \ |    \ |  |
| |    |  o  ||  o  ) /  [_ | |     |  | |  _  ||  o  ||      ||     ||  D  )|  |
| |___ |     ||     ||    _]| |___  |  | |  |  ||     ||_|  |_||  O  ||    / |__|
|     ||  _  ||  O  ||   [_ |     | |  | |  |  ||  _  |  |  |  |     ||    \  __ 
|     ||  |  ||     ||     ||     | |  | |  |  ||  |  |  |  |  |     ||  .  \|  |
|_____||__|__||_____||_____||_____||____||__|__||__|__|  |__|   \___/ |__|\_||__|
                                                                                
"""
)

setup_logging()

#=== Config
WINDOW_NAME = "EVOC labeler"
HELP_WINDOW = "Help Instructions" 
CIRCLE_SIZE = 3

#=== Typedef
class Args(NamedTuple):
    indir:           str
    outdir:          str
    markdir:         str
    ext:             str
    contribute_mode: bool
    ordered:         bool

class Vector2(NamedTuple):
    x: int = 0
    y: int = 0

@dataclass
class SampleLabels():
    """A record of all positions found within a sample"""
    head:   list[Vector2] = field(default_factory=lambda: [])
    boxes:  list[Vector2] = field(default_factory=lambda: [])
    joints: list[Vector2] = field(default_factory=lambda: []) 

@dataclass
class ColorSet():
    blue  = (0, 255, 0)
    red   = (0, 0, 255)
    white = (255, 255,255)

@dataclass
class SampleLabelColorSet():
    head  = ColorSet.white
    joint = ColorSet.blue
    box   = ColorSet.red


@dataclass
class MutableVector2():
    """Pointer wrapper for a vector2 tuple type"""
    v: Vector2

args: Args = parser.parse_args()

#=== Event Handlers. One for mouse capture the other for keyboard capture
def on_mousemove(record_out: MutableVector2):
    def update_position(event: int, x: int, y: int, flags, param):
        if(not event == cv2.EVENT_MOUSEMOVE): return
        record_out.v = Vector2(x=x, y=y)

    cv2.setMouseCallback(WINDOW_NAME, update_position)

def on_keypress(event: Callable[int, bool]):
    """
    Blocking function that allows for reading of events from the keyboard in cv2
    """
    # Continuously listen for keypresses until callback returns false
    while True:
        key = cv2.waitKey(1) & 0xFF
        if not event(key): return

#=== Yihong Training Compatability Layer
def save_json(record: SampleLabels, file: str, img: cv2.UMat):

    # For some reason we need to set all these points to zero in location
    # I do not know why. Contact Yihong as this is the compatibility layer 
    # between his training scripts and our labelling scripts    
    nullify = lambda x: [[0,0,0] for i in range(len(x))]

    sample = {
        "objects": [{
            "class": "gecko",
            "keypoints": [
                {"name": "head", "location": nullify(record.head), "projected_location": record.head},
                {"name": "joint", "location": nullify(record.joints), "projected_location": record.joints},
                {"name": "box", "location": nullify(record.boxes), "projected_location": record.boxes}
            ]
        }]
    }

    logging.info("Collected label for image")
    logging.info(f"\n{pformat(sample)}")

    with open(os.path.join(args.outdir, file.replace(args.ext, '.json')), 'w') as buf:
        buf.write(json.dumps(sample, indent=4))
    
    # The layer also required a direct copy to be made of the image. So we will 
    # copy directly from the `indir` into `outdir`
    cv2.imwrite(os.path.join(args.outdir, file), img)

#=== Image Processor
def label_image(file: str) -> None:
    logging.info(f"Starting labeler for file: {file}")

    sample = SampleLabels()

    # This value is mutated by `on_mousemove`. It contains the current
    # coordinates of the mouse. Is asynchronously updated on the same thread
    # so in theory thread-safe. NOTE: READ ONLY
    position = MutableVector2(Vector2())
    mut_last_keypress: int = -1

    img = cv2.imread(os.path.join(args.indir, file))
    img_initial_state = img.copy()

    def capture_keypress(key: int) -> bool:
        # Python find therapy please. It's just the scope above
        nonlocal img, img_initial_state, sample, position, mut_last_keypress

        mut_last_keypress = key

        # Capture (h)ead, (j)oint, (b)ox respectively.
        if key == ord('h'): 
            sample.head.append(Vector2(*position.v))
            cv2.circle(
                img, (position.v.x, position.v.y), 
                CIRCLE_SIZE, SampleLabelColorSet.head, -1
            )

        if key == ord('j'): 
            sample.joints.append(Vector2(*position.v))
            cv2.circle(
                img, (position.v.x, position.v.y), 
                CIRCLE_SIZE, SampleLabelColorSet.joint, -1
            )
 
        if key == ord('b'): 
            sample.boxes.append(Vector2(*position.v))
            cv2.circle(
                img, (position.v.x, position.v.y), 
                CIRCLE_SIZE, SampleLabelColorSet.box, -1
            )
        
        if key == ord('h') or key == ord('j') or key == ord('b'):
            logging.info(f"Recored image coordinate: {position.v}")
 
        if key == ord('r'):
            logging.info(f"Resetting labels")
            img = img_initial_state.copy()
            sample = SampleLabels()
        
        # Show updated image and write update to the marked image file
        cv2.imshow(WINDOW_NAME, img)

        if key == ord('q'): 
            cv2.destroyAllWindows()
            exit()

        # `.` and `d` specifies to go next image. return False to tell the mouse
        # capture function we are done with this image
        if key == ord('.') or key == ord('d'): return False
 
        return True

    cv2.putText(img, file.split('.')[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    started_at = time() 
    cv2.imshow(WINDOW_NAME, img)

    on_mousemove(position)
    on_keypress(capture_keypress)

    if mut_last_keypress == ord('d'):
        logging.info("Sample discarded")
        return

    # After collection is complete, save the collected labels out including the
    # edited image
    cv2.imwrite(os.path.join(args.markdir, file), img)
    save_json(sample, file, img_initial_state)

    logging.info(f"Label took {time() - started_at:.2f} seconds!")

    total, so_far, percent = get_labeling_progress(args.indir, args.markdir)
    logging.info(
        f"\n{so_far}/{total}\
          \nLabeling Progress: {percent}%\
          \nLabeling took {time() - started_at:.2f} seconds!")


#=== Script Begins Here!
# Create the output folders if DNE
os.makedirs(args.outdir, exist_ok=True)
os.makedirs(args.markdir, exist_ok=True)

# Validate the dataset directory
dataset = os.listdir(args.indir)
if(len(dataset) == 0): 
    raise Exception(
        f"Error: Dataset is empty! Dataset should be populated in folder: {args.indir}")


# If the user wants them ordered, sort the files by ID. 
# If not, we randomize what samples we do.
if args.ordered:
    logging.info("ARG: Sorted labeling is active")
    dataset.sort(key=lambda x: int(x.split('.')[0]))
else:
    logging.info("ARG: Randomized labeling is active")
    shuffle(dataset)    

# Contribute mode removes files from the labeler that are already labeled. 
# If this is disabled, you will potentially be overwriting already labeled files
# and relabeling them.
if args.contribute_mode:
    logging.warning(
        "ARG: CONTRIBUTE MODE IS ACTIVE. All files already labeled will be automatically skipped")

    # Filter out files already done
    dataset = filter(
        lambda x: not os.path.exists(os.path.join(args.outdir, x)), dataset) 

# Load the keyboard cheatsheet graphic
try:
    usage = cv2.imread("usage.png")
    height, width = usage.shape[:2]
    usage_resize = cv2.resize(usage, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(HELP_WINDOW, usage_resize)
except:
    logging.warning("Did not start script in same directory. Could not load usage.png")

# Main Loop, label each image
for img in dataset: label_image(img)

logging.info("Dataset is completely labeled! Exiting")
cv2.destroyAllWindows()
