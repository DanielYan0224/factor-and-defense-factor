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





#%%







#%%





#%%