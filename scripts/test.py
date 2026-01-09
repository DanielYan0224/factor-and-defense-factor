#%%

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

#%%

data_dir = '/neodata/open_dataset/mlb_data/preprocessed'
filename = 'truncated_data_with_rtheta_team.parquet'
#filename = 'rtheta_prob_tbl.parquet'
df = pd.read_parquet(os.path.join(data_dir, filename))

#%%

df.info()

#%%

import json

class Config:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self, key, value)
        
        


#%%





#%%







#%%





#%%