#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

from calculate_score import combined_score_tbl
from expect_score import get_truncated_dataset_with_team

df = get_truncated_dataset_with_team()

dp(df. head(5))
#%%
nyy_batter_df = df[
  (df['game_year'] == 2014)&
  (df)
]

#%%