#%%
# Standard Library
from pathlib import Path

# Data Science
import pandas as pd
import numpy as np

# Visualization
from IPython.display import display as dp

# Domain / Third Party
from pybaseball import team_fielding, team_ids, fielding_stats


pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

df = pd.read_parquet(DATA_PROCESSED / "truncated_data_with_rtheta_team.parquet")

df = df[df['game_type'] == 'R']

#%%
tm = 'NYY'
team_df = df[df['home_team'] == tm]

dp(team_df.columns)
#%%
slg_weights = {
    'single': 1, 'double': 2, 'triple': 3, 'home_run': 4,
}

# 1. Target Team 在 Target Park (主場進攻)
mask_1 = (team_df['batter_team'] == tm) & (team_df['home_team'] == tm)
    
# 2. Target Team 在 Other Parks (客場進攻)
mask_2 = (team_df['batter_team'] == tm) & (team_df['home_team'] != tm)
    
# 3. Other Teams 在 Target Park (客隊在目標球場進攻)
mask_3 = (team_df['batter_team'] != tm) & (team_df['home_team'] == tm)
    
# 4. Other Teams 在 Other Parks (聯盟平均環境)
mask_4 = (team_df['batter_team'] != tm) & (team_df['home_team'] != tm)
#%%