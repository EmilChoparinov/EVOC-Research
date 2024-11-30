#!/bin/bash

# DTW    < .... > DISTANCE
# Alpha: 0      1

STANDARD_RUNS=2
STANDARD_GENS=2
STANDARD_ANIM=model/slow_with_linear_4.csv
STANDARD_VAES=model/vae_model.pth
STANDARD_SIMS=DTW

idemp_move() {
  dir="$1"

  rm -rf "$dir"
  mkdir "$dir"
  mv *.csv "$dir"
  mv *.txt "$dir"
}

echo "Running Alpha 0: Similarity Only"
python3 run.py --cleanup \
  --animal-data model/slow_with_linear_4.csv \
  --vae model/vae_model.pth \
  --gens "$STANDARD_GENS" --runs "$STANDARD_RUNS" --alpha 0 --similarity-type "$STANDARD_SIMS"
idemp_move "dtw-0-similarity"

echo "Running Alpha 0.25"
python3 run.py --cleanup \
  --animal-data model/slow_with_linear_4.csv \
  --vae model/vae_model.pth \
  --gens "$STANDARD_GENS" --runs "$STANDARD_RUNS" --alpha 0.25 --similarity-type "$STANDARD_SIMS"
idemp_move "dtw-0.25"

echo "Running Alpha 0.5"
python3 run.py --cleanup \
  --animal-data model/slow_with_linear_4.csv \
  --vae model/vae_model.pth \
  --gens "$STANDARD_GENS" --runs "$STANDARD_RUNS" --alpha 0.5 --similarity-type "$STANDARD_SIMS"
idemp_move "dtw-0.5"

echo "Running Alpha 0.75"
python3 run.py --cleanup \
  --animal-data model/slow_with_linear_4.csv \
  --vae model/vae_model.pth \
  --gens "$STANDARD_GENS" --runs "$STANDARD_RUNS" --alpha 0.75 --similarity-type "$STANDARD_SIMS"
idemp_move "dtw-0.75"

echo "Running Alpha 1: Distance Only"
python3 run.py --cleanup \
  --animal-data model/slow_with_linear_4.csv \
  --vae model/vae_model.pth \
  --gens "$STANDARD_GENS" --runs "$STANDARD_RUNS" --alpha 1 --similarity-type "$STANDARD_SIMS"
idemp_move "dtw-1-distance"