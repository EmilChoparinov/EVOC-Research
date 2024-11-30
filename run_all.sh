
# Alpha 0: Similarity
python3 -m simulate.run --gens 300 --runs 30 --alpha 0 --with-fit similarity
# Alpha 0.25: Blended
# Alpha 0.50: Blended
# Alpha 0.75: Blended
# Alpha 1: Distance
python3 -m simulate.run --gens 300 --runs 30 --alpha 1 --with-fit distance


python src/run.py --cleanup --runs 1 --gens 200 --alpha 0.25 --similarity_type DTW