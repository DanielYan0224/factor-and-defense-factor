#%%
import pandas as pd
import os
from pathlib import Path

from IPython.display import display as dp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import joblib 

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

# 路徑設定
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
CACHE_DIR = DATA_PROCESSED / "_cache"

# 確保快取目錄存在
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = DATA_PROCESSED / "truncated_data_with_rtheta.parquet"
CACHE_PATH = CACHE_DIR / "savant_data_cache.pkl"
PROB_TBL_PATH = DATA_PROCESSED / "rtheta_prob_tbl.parquet"
OUTPUT_PATH = DATA_PROCESSED / "truncated_data_with_rtheta_team.parquet"

# # 取得所有年份並排序
# years = sorted(df['game_year'].unique())

# # 建立一個字典：{2015: {'NYY', 'BOS', ...}, 2016: {...}}
# teams_by_year = {y: set(df[df['game_year'] == y]['home_team'].unique()) for y in years}

# # 兩兩比對
# for i in range(len(years) - 1):
#     y1 = years[i]
#     y2 = years[i+1]
    
#     set1 = teams_by_year[y1]
#     set2 = teams_by_year[y2]
    
#     # 消失的隊伍 (在 y1 有，但 y2 沒了)
#     disappeared = set1 - set2
#     # 新增的隊伍 (在 y2 有，但 y1 沒有)
#     new_added = set2 - set1
    
#     if disappeared or new_added:
#         print(f"--- {y1} -> {y2} 發生變化 ---")
#         if disappeared:
#             print(f"  消失: {disappeared}")
#         if new_added:
#             print(f"  新增: {new_added}")

def get_whole_dataset():
    """回傳完整的 Parquet 主資料集"""
    path = DATA_PROCESSED / "savant_data_15_24.parquet"
    if not path.exists():
        raise FileNotFoundError(f"找不到完整資料集：{path}")
    print(f"載入完整資料：{path}")
    return pd.read_parquet(path)


def get_truncated_dataset():
    """回傳只含主要欄位的 truncated 資料"""
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"找不到 truncated 資料：{PARQUET_PATH}")
    print(f"📦 載入 truncated 資料：{PARQUET_PATH}")
    return pd.read_parquet(PARQUET_PATH)


def get_rtheta_prob_tbl():
    """回傳 r_theta 對應的事件機率分佈表"""
    if not PROB_TBL_PATH.exists():
        raise FileNotFoundError(
            f"找不到機率表：{PROB_TBL_PATH}\n請先執行 expect_score.py 產生它。"
        )
    print(f"載入 r_theta 機率表：{PROB_TBL_PATH}")
    return pd.read_parquet(PROB_TBL_PATH)


def assign_pitcher_batter_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    根據每一列的 inning_topbot 欄位，指派 pitcher_team 與 batter_team。
    """
    df = df.copy()
    
    # Vectorized assignment
    is_top = df['inning_topbot'] == 'Top'
    df['pitcher_team'] = np.where(is_top, df['home_team'], df['away_team'])
    df['batter_team'] = np.where(is_top, df['away_team'], df['home_team'])
    
    # 調整欄位順序：將 pitcher/batter team 放到 away_team 後面
    cols = list(df.columns)
    if 'away_team' in cols:
        insert_pos = cols.index('away_team') + 1
        for col in ['pitcher_team', 'batter_team']:
            if col in cols:
                cols.remove(col)
            cols.insert(insert_pos, col)
            insert_pos += 1
            
    return df[cols]


def get_truncated_dataset_with_team():
    """回傳 tuncated data with rtheta and batter/pitcher team"""
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"找不到 truncated 資料：{OUTPUT_PATH}")
    print(f"📦 載入 truncated 資料：{OUTPUT_PATH}")
    return pd.read_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    # 1. 讀取資料 (優先讀取快取)
    if CACHE_PATH.exists():
        print("從快取讀取主資料中...")
        df = joblib.load(CACHE_PATH)
    else:
        print("讀取 parquet 並建立快取...")
        df = get_truncated_dataset()
        joblib.dump(df, CACHE_PATH)
        print(f"已建立快取：{CACHE_PATH}")
    # check and replace team name
    print("正在統一球隊代號 (ATH -> OAK)...")
    # 定義需要檢查的欄位 (通常原始資料會有 home_team, away_team)
    # 如果這時候還沒有 pitcher_team/batter_team，list 裡寫了也不會報錯 (因為有檢查 if col in df.columns)
    team_cols = ['home_team', 'away_team', 'pitcher_team', 'batter_team']
    
    for col in team_cols:
        if col in df.columns:
            # 使用 replace 將 ATH 替換為 OAK
            # 如果資料量很大，這樣寫效率不錯
            mask = df[col] == 'ATH'
            if mask.any():
                print(f"  - 修正欄位 {col}: {mask.sum()} 筆")
                df.loc[mask, col] = 'OAK'
                
    # 2. 建立每一個 r theta of hip 的 events 的 機率 table
    print("計算 r_theta 機率表...")
    rtheta_prob_tbl = df.groupby('r_theta')['events'].value_counts(normalize=True).reset_index()
    rtheta_prob_tbl.columns = ['r_theta', 'events', 'probability']
    rtheta_prob_tbl.to_parquet(PROB_TBL_PATH)
    print(f"已儲存機率表：{PROB_TBL_PATH}")

    # 3. 建立 batter team and pitcher team 的 cols 並存檔
    print("處理隊伍指派...")
    df_teams = assign_pitcher_batter_teams(df)
    df_teams.to_parquet(OUTPUT_PATH)
    print(f"已儲存最終資料：{OUTPUT_PATH}")
# %%
