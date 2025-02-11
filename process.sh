python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0 --similarity-type DTW --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.25 --similarity-type DTW --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.5 --similarity-type DTW --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.75 --similarity-type DTW --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 1 --similarity-type DTW --runs 1 --gens 300

python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0 --similarity-type MSE --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.25 --similarity-type MSE --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.5 --similarity-type MSE --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.75 --similarity-type MSE --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 1 --similarity-type MSE --runs 1 --gens 300

python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0 --similarity-type Angles --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.25 --similarity-type Angles --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.5 --similarity-type Angles --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 0.75 --similarity-type Angles --runs 1 --gens 300
python3 -m simulate.main --animal-data simulate/model/slow_with_linear_4.csv --alpha 1 --similarity-type Angles --runs 1 --gens 300
