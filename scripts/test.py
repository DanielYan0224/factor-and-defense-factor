#%%

import pandas as pd
import os
from utils import Config, get_expected_bases_map, prepare_regression_data
import statsmodels.formula.api as smf
import numpy as np

data_dir = '/neodata/open_dataset/mlb_data/preprocessed'
prob_table_filename = 'rtheta_prob_tbl.parquet'
input_filename = 'truncated_data_with_rtheta_team.parquet'
input_file_path = os.path.join(data_dir, input_filename)
df = pd.read_parquet(input_file_path)
config = Config()
event_weights = config.weights
exp_map = get_expected_bases_map(config=config)
reg_df = prepare_regression_data(df, exp_map, config)

year = 2015

data_year = reg_df[reg_df['game_year'] == year].copy()
model = smf.wls("avg_residual ~ C(park) + C(defense)", data=data_year, weights=data_year['weight'])
res = model.fit()
params = res.params

all_parks = sorted(data_year['park'].unique())
all_defenses = sorted(data_year['defense'].unique())

beta_park_raw = {} 
beta_def_raw = {} 
intercept_raw = params['Intercept']

for p in all_parks:
    key = f"C(park)[T.{p}]"
    beta_park_raw[p] = params.get(key, 0.0)
        
for d in all_defenses:
    key = f"C(defense)[T.{d}]"
    beta_def_raw[d] = params.get(key, 0.0)

mean_park = np.mean(list(beta_park_raw.values()))
mean_def = np.mean(list(beta_def_raw.values()))

beta_park_centered = {k: v - mean_park for k, v in beta_park_raw.items()}
beta_def_centered = {k: v - mean_def for k, v in beta_def_raw.items()}
adj_intercept = intercept_raw + mean_park + mean_def

std_park = np.std(list(beta_park_centered.values()))
std_def = np.std(list(beta_def_centered.values()))


park_indices = {}
for k, v in beta_park_centered.items():
    z = v / std_park if std_park > 0 else 0
    park_indices[k] = 100 + 20 * z

defense_indices = {}
for k, v in beta_def_centered.items():
    z = v / std_def if std_def > 0 else 0
    defense_indices[k] = 100 - 20 * z

#%%

defense_indices



#%%





#%%







#%%




#%%





#%%





#%%