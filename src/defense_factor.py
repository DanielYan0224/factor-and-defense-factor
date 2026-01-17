#%%
import pandas as pd
import numpy as np
from IPython.display import display as dp

from pybaseball import team_fielding, team_ids

from pathlib import Path

from expect_score import get_truncated_dataset_with_team
pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

df_defense = pd.read_parquet(DATA_RAW / "defense_data.parquet")
df_truncated = get_truncated_dataset_with_team()

# dp(df_truncated['home_team'].unique())
# dp(len(df_defense['Team'].unique()))

def transform_team_name(df: pd.DataFrame, 
                        mapping_dict: dict[str, str], 
                        target_col: str = "Team") -> pd.DataFrame:
    """
    轉換 DataFrame 中的隊名，並檢查是否有遺漏的對應。
    
    Args:
        df: 要處理的 DataFrame
        mapping_dict: 隊名轉換字典
        target_col: 要轉換的欄位名稱 (預設為 'Team')
    
    Returns:
        處理過後的 DataFrame
    """
    df_result = df.copy()
    
    unique_teams = df_result[target_col].unique()
    
    # compare
    missing_teams = [team for team in unique_teams if team not in mapping_dict]

    # report the missing teams
    if len(missing_teams) > 0:
        print(f"⚠️ 警告：欄位 '{target_col}' 中有 {len(missing_teams)} 個隊名不在字典中：")
        print(missing_teams)
        print("--> 這些隊伍轉換後將變成 NaN，請檢查字典。")
    else:
        print(f"✅ 檢查通過！欄位 '{target_col}' 所有隊名皆在字典中。")

    # 4. 進行轉換 (使用 map)
    # 這裡的邏輯是：如果有對應就轉，沒對應就變成 NaN (因為使用了 map)
    df_result[target_col] = df_result[target_col].map(mapping_dict)
    
    return df_result



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


df_v2 = transform_team_name(df=df_defense, 
                            mapping_dict=team_name_transfer_dict, 
                            target_col='Team')

df_v2.to_parquet(DATA_PROCESSED / "defense_data_team_transfer.parquet", index=False)
#%%
defense_data = team_fielding(2015, 2024, ind=1)

def_col_mask = ['FP', 'Def', 'DRS', 'UZR', 'OAA', 'UZR/150', 'RngR', 'Range']

mask = ['Season', 'Team'] + def_col_mask 

defense_truncated_data = defense_data[mask]

#%%