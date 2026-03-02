#%%
# Standard Library
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Data Science
import pandas as pd
import numpy as np

# Visualization
from IPython.display import display as dp

pd.set_option('display.max_rows', None)  # 顯示所有行
pd.set_option('display.max_columns', None)  # 顯示所有列
pd.set_option('display.width', None)  # 自動調整寬度以適應內容
pd.set_option('display.max_colwidth', None)  # 不限制單個列的最大寬度
 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

year = 2024

file_path = DATA_PROCESSED / "savant_data_15_24.parquet"


df = pd.read_parquet(file_path)

df_games = df[(df['game_type'] == 'R')  
                & (df['game_year'] == year)].drop_duplicates('game_pk')


# 2. 把主場隊伍和客場隊伍合併成一個長數列
all_teams = pd.concat([df_games['home_team'], df_games['away_team']])

# 3. 計算出現次數並轉換成 DataFrame
team_counts = all_teams.value_counts().reset_index()

# 4. 重新命名欄位
team_counts.columns = ['team', 'num_game']

team_counts['year'] = year

# 按照字母排序，並重排欄位順序 (讓 year 在最前面)
team_counts = team_counts.sort_values('team')[['year', 'team', 'num_game']]

dp(team_counts)

#%%
# 檢查 HOU 每一年的場次
hou_games_count = df[df['home_team'] == 'HOU'].groupby('game_year').size()
print(hou_games_count)
#%%