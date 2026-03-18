## How to install
### Requirements: 
An Anaconda python environment is recommend.Check the environment.yml file, but primarily:
Python >= 3.5
numpy
pandas
sklearn
xgboost
imbalanced-learn
joblib


## Run
grid search:
```
python run.py --model Random Forest --grid_search True
```
use params:

`python run.py --model Random Forest`

## Output File

dataset split file

`/model_results/split_data`

model predict result file

`/model_results/result`


