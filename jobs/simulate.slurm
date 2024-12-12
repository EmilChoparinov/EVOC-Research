#!/bin/bash
#SBATCH --job-name=run_experiments
#SBATCH --output=experiments_%j.log
#SBATCH --error=experiments_%j.err
#SBATCH --time=4-00:00:00             
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emilchoparinov@gmail.com
#SBATCH --job-name="DTW - EVOC Group 7"

# Activate Python environment
source ../enter.sh

OUTPUT_ROOT="~/dtw-experiments"

STANDARD_PATH="../simulate/run.py"
STANDARD_RUNS=30
STANDARD_GENS=300
STANDARD_ANIM="../simulate/model/slow_with_linear_4.csv"
STANDARD_VAES="../simulate/model/vae_model.pth"
STANDARD_SIMS=DTW

mkdir -p "$OUTPUT_ROOT"
cd "$OUTPUT_ROOT"

new_experiment() {
  dir="$1"

  rm -rf "$dir" 2>/dev/null
  mkdir -p "$dir"
  cd "$dir"
}

run_experiment() {
  local alpha="$1"
  local exp_dir="dtw-${alpha//./}"
  
  echo "Running Alpha $alpha"
  new_experiment "$exp_dir"

  echo "Boot up experiment"  
  python3 "$STANDARD_PATH" --cleanup \
    --animal-data "$STANDARD_ANIM" \
    --vae "$STANDARD_VAES" \
    --gens "$STANDARD_GENS" --runs "$STANDARD_RUNS" \
    --alpha "$alpha" --similarity-type "$STANDARD_SIMS"
  
  cd ..
}

alphas=(0 0.25 0.5 0.75 1)
run_experiment "${alphas[$SLURM_ARRAY_TASK_ID]}"

# Run experiments
#srun -N1 -n1 run_experiment 0 
#srun -N1 -n1  run_experiment 0.25
#srun -N1 -n1  run_experiment 0.5
#srun -N1 -n1  run_experiment 0.75
#srun -N1 -n1  run_experiment 1
#wait

echo "Distributed tasks completed."
