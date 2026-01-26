#%%
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
from utils import classify_pull_oppo, analyze_spray_distribution, assign_pitcher_batter_teams

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"


df = pd.read_parquet(DATA_PROCESSED / "savant_data_15_24.parquet")




df = df.dropna(subset=['hc_x', 'hc_y']).copy()
df = df[(df['description'] == 'hit_into_play') & 
    (df['game_type'] == 'R')]

df = assign_pitcher_batter_teams(df)

drop_events = [
    'sac_bunt',              # 短打
    'sac_bunt_double_play',  # 短打雙殺
    'catcher_interf',        # 捕手妨礙 (非擊球)
    'game_advisory',         # 系統雜訊
    'sac_fly',               # 犧牲飛球
    'sac_fly_double_play',   # 犧牲飛球雙殺
    #'field_error',         
    None                     # 空值
]

# 執行篩選：只保留 "不在" 清單內的資料
df_clean = df[~df['events'].isin(drop_events)].copy()

df_clean = analyze_spray_distribution(classify_pull_oppo(df_clean))

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(DATA_PROCESSED / "spray_data_15_24.csv")

# 1. 確保有 'game_year' 欄位 (通常 pybaseball 會有，如果沒有就從 game_date 抓)
if 'game_year' not in df.columns:
    df['game_year'] = pd.to_datetime(df['game_date']).dt.year

# 2. 篩選目標球隊
target_teams = ['BOS', 'NYY', 'PIT', 'SF']
df_target = df[df['home_team'].isin(target_teams)].copy()

# 3. 分組計算數量 (Count)
# Group By: 球隊, 年份, 左右打, 擊球方向
stats = df_target.groupby(
    ['home_team', 'game_year', 'stand', 'batted_ball_direction', 'bb_type']
).size().reset_index(name='count')

# 4. 計算比例 (Percentage)
# 算出每個 "球隊-年份-左右打" 的總球數 (分母)
stats['total'] = stats.groupby(['home_team', 'game_year', 'stand'])['count'].transform('sum')

# 算出百分比
stats['pct'] = (stats['count'] / stats['total']) * 100

dp(stats['home_team'].unique())
#%%

# 5. 整理表格顯示 (Format)
# 讓我們把表格變漂亮一點，只看我們關心的 Pull/Oppo/Center
# 這裡示範顯示前幾筆
print("=== 每年擊球方向數據表 (部分) ===")
dp(stats[stats['home_team'] == 'BOS'])

#%%