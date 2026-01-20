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

# Local Modules
from expect_score import get_truncated_dataset_with_team
from utils import team_name_transfer_dict, transform_team_name, filter_defense_data

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

defense_df = pd.read_parquet(DATA_RAW / "defense_data.parquet")

team_mapping = team_name_transfer_dict

defense_transfer_data = transform_team_name(df = defense_df, 
                        mapping_dict = team_mapping,
                        target_col = "Team")

defense_transfer_data.to_csv(DATA_PROCESSED / "defense_data_transfer.csv", index=False)
#%%
defense_transfer_data = pd.read_csv(DATA_PROCESSED / "defense_data_transfer.csv")
dp(defense_transfer_data.head())


#%%
defense_transfer_data.to_csv(DATA_PROCESSED / "defense_data_transfer_truncated.csv", index=False)

def_cols = ['FP', 'Def', 'DRS', 'UZR', 'OAA', 'UZR/150', 'RngR', 'Range']

defense_transfer_truncated_data = filter_defense_data(df = defense_transfer_data, 
                                                    target_cols = def_cols)

defense_transfer_truncated_data.to_csv(DATA_PROCESSED / "defense_data_transfer_truncated.csv", index=False)
#%%
defense_player_df = fielding_stats(2023, qual=None)

def_cols = ['FP', 'Def', 'DRS', 'UZR', 'OAA', 'UZR/150', 'RngR', 'Range']

defense_pl_truncated_df = defense_player_df[['Name', 'Team', 'Season'] + def_cols]
dp(defense_pl_truncated_df.sample(10))
#%%
from pybaseball import statcast_outs_above_average

# All fielders with at least 50 fielding attempts in 2019
data = statcast_outs_above_average(2023, "all", 0)
df = data[data['player_id'] == 657557]

dp(df)
#%%