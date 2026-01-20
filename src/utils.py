# Standard Library
from pathlib import Path

# Data Science
import pandas as pd
import numpy as np

# Visualization
from IPython.display import display as dp

team_name_transfer_dict = {
    # 美聯東區
    'Blue Jays': 'TOR', 
    'Orioles': 'BAL', 
    'Rays': 'TB',
    'Red Sox': 'BOS',
    'Yankees': 'NYY',

    # 美聯中區
    'Guardians': 'CLE', 'Indians': 'CLE', 'Cleveland': 'CLE',
    'Royals': 'KC', 
    'Tigers': 'DET', 
    'Twins': 'MIN',
    'White Sox': 'CWS',

    # 美聯西區
    'Angels': 'LAA', 
    'Astros': 'HOU',
    'Athletics': 'OAK',
    'Mariners': 'SEA',
    'Rangers': 'TEX',

    # 國聯東區
    'Braves': 'ATL',
    'Marlins': 'MIA',
    'Mets': 'NYM',
    'Nationals': 'WSH',
    'Phillies': 'PHI',

    # 國聯中區
    'Brewers': 'MIL', 
    'Cardinals': 'STL', 
    'Cubs': 'CHC', 
    'Pirates': 'PIT',               
    'Reds': 'CIN', 

    # 國聯西區
    'Diamondbacks': 'ARI', 
    'Dodgers': 'LAD',
    'Giants': 'SF',
    'Padres': 'SD',
    'Rockies': 'COL'
}

def transform_team_name(df: pd.DataFrame, 
                        mapping_dict: dict[str, str], 
                        target_col: str = "Team") -> pd.DataFrame:
    """
    轉換 DataFrame 中的隊名，檢查是否有遺漏的對應，並將 'Season' 欄位更名為 'game_year'。
    
    Args:
        df: 要處理的 DataFrame
        mapping_dict: 隊名轉換字典
        target_col: 要轉換的欄位名稱 (預設為 'Team')
    
    Returns:
        處理過後的 DataFrame (隊名已轉換，且 'Season' 欄位已改為 'game_year')
    """
    df_result = df.copy()
    
    unique_teams = df_result[target_col].unique()
    
    # compare
    missing_teams = [team for team in unique_teams if team not in mapping_dict]

    # report the missing teams
    if len(missing_teams) > 0:
        print(f"Warning: column '{target_col}' has {len(missing_teams)} teams that are not in the dictionary:")
        print(missing_teams)
        print("These teams will become NaN after conversion. Please check the dictionary.")
    else:
        print(f"Check passed! All teams in column '{target_col}' are in the dictionary.")

    # 4. 進行轉換 (使用 map)
    # 這裡的邏輯是：如果有對應就轉，沒對應就變成 NaN (因為使用了 map)
    df_result[target_col] = df_result[target_col].map(mapping_dict)
    
    # 5. 把 column: Season 轉成 column: game_year
    df_result = df_result.rename(columns={"Season": "game_year"})
    
    return df_result

def filter_defense_data(df: pd.DataFrame, 
                        target_cols: list = None) -> pd.DataFrame:
    """
    篩選防守數據欄位的函式。
    
    參數:
    df (pd.DataFrame): 原始的 DataFrame。
    target_cols (list): 你想要篩選的防守欄位列表。
                        如果不填，會使用預設的欄位清單。
    
    回傳:
    pd.DataFrame: 篩選完成的 DataFrame。
    """
    
    # 1. 設定預設的防守欄位清單
    if target_cols is None:
        target_cols = ['FP', 'Def', 'DRS', 'UZR', 'OAA', 'UZR/150', 'RngR', 'Range']
    
    # 2. 設定固定要保留的基礎欄位 (例如年份跟隊伍)
    base_cols = ['game_year', 'Team']
    
    # 3. 合併所有需要的欄位
    # 注意：這裡做一個檢查，確保不會重複加入欄位
    final_mask = base_cols + [col for col in target_cols if col not in base_cols]
    
    # 4. 回傳篩選後的資料
    # 使用 .copy() 是好習慣，避免之後修改跳出 SettingWithCopyWarning 警告
    return df[final_mask].copy()