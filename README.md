# Evolutionary Computing @ VU 2024
This repository contains code for performing simulations, gathering data, and training for the 2024 CI-EVOC Project 7

## Contents Guide
There are multiple directories. 
- `/src` contains the simulation training code
- `/apps` contains python apps that interact with the simulation or the physical robot

## Prerequisites
- A valid version of python required. I am under `3.10.12`. You can check by doing `python3 --version`. NOTE: `Python 3.12.x` do not work. `Revolve2-standards` requires either 3.10.x or 3.11.x.

- A dependency attached to this repository from the CI-Group, MultiNEAT is required for installing
  Revolve2. You can get it on Linux or Mac with the following commands:
```
# pacman
sudo pacman -S cereal

# apt
sudo apt install cereal
```

- Compatible with: Linux, MacOS Sequoia, Windows 10/11 (Under WSL)

## Usage
Run the `enter.sh` script. It will install the revolve2 environment for you. It will activate for you as well. When interacting with the codebase, always do:

```
source enter.sh
```

If there are issues, you can manually follow the [revolve2 setup guide](https://ci-group.github.io/revolve2/installation/index.html#prerequisites) instead.

If you have driver errors please contact me. You most likely do not have CUDA being detected by python correctly and I could help.

## Running the EA
`src/run.py` is the entry point. You can run it as is or use on of the options to customize:

```
usage: run.py [-h] [--cleanup] [--skip] [--runs RUNS] [--gen GEN]

options:
  -h, --help   show this help message and exit
  --cleanup    Delete *.csv, *.txt in current directory
  --skip       Do not perform ea iterations
  --runs RUNS  Times to run EA
  --gens GEN    How many generations per run
```

### Helpful Configurations:
Clears the directory of all CSVs and log files.
```
python3 run.py --cleanup --skip
```

Ensures that each run has new files to write to before starting EA.
```
python3 run.py --cleanup
```

## Apps
The apps folder contains python files that may be invoked with generated data to
do different behaviors:
- `evaluate.py` Performs an evaluation on a given genotype and no more
- `remote_connection_test.py` Tests if robot can connect remotely and move something
- `simulate.py` Play the simulation on a given genotype and no more 

## Important Configurations
Emil and Yushuang have setup the I/O pins configured as:

```
Left Arm: 0
Left Leg: 1
Torso: 8
Reft Arm: 31
Relf Leg: 30
Tail: 24
```

Use the above legend as a reference when editing physical robot related code.

## For the `apps.game_sim` App
You need to also additionally install pygame. Do `python3 -m pip install pygame` within your virtual
environment.

## Contacts
Emil Choparionv - emilchoparinov@gmail.com 
Yushuang Wang