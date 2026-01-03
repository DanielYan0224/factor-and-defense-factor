#%%

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

#%%

data_dir = '/neodata/open_dataset/mlb_data/preprocessed'
parquet_filename = 'rtheta_prob_tbl.parquet'
file_path = os.path.join(data_dir, parquet_filename)
df = pd.read_parquet(file_path)
df.head()

#%%

df.info()

#%%




#%%








#%%





#%%







#%%





#%%