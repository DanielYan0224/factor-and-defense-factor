#%%
import pandas as pd
import numpy as np
from IPython.display import display as dp

from pybaseball import team_fielding, team_batting

from pathlib import Path

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
df = pd.read_parquet(DATA_PROCESSED / "truncated_data_with_rtheta_team.parquet")


defense_data = team_fielding(2015, 2024, ind=1)

def_col_mask = ['FP', 'Def', 'DRS', 'UZR', 'OAA', 'UZR/150', 'RngR', 'Range']

mask = ['Season', 'Team'] + def_col_mask 

defense_truncated_data = defense_data[mask]

dp(defense_data.sample(5, random_state=42))
dp(defense_truncated_data.sample(5, random_state=42))
#%%