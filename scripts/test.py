#%%

import pandas as pd
import os

#%%

data_dir = '/neodata/open_dataset/mlb_data'

os.listdir(data_dir)

all_data = []

for year in range(2014, 2025):
    file_path = os.path.join(data_dir, f'statcast_{year}.csv')
    df = pd.read_csv(file_path)
    df['year'] = year  # Add year column to avoid confusion
    all_data.append(df)

#%%







#%%










#%%







#%%










#%%







#%%