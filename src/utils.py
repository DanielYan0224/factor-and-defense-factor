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

def classify_pull_oppo(df):
    """
    輸入: 包含 'hc_x', 'hc_y', 'stand' 的 DataFrame
    輸出: 增加 'spray_angle' 和 'batted_ball_direction' 欄位
    """
    
    # --- 步驟 1: 先算出噴灑角度 (Spray Angle) ---
    # 0度=中外野, 負值=左外野, 正值=右外野
    hc_x_origin = 125.42
    hc_y_origin = 198.27
    df['location_x'] = (df['hc_x'] - hc_x_origin) * 2.495
    df['location_y'] = (hc_y_origin - df['hc_y']) * 2.495
    df['spray_angle'] = np.degrees(np.arctan2(df['location_x'], df['location_y']))
    
    # --- 步驟 2: 定義判斷標準 (Thresholds) ---
    # 業界標準通常定為 15 度 或 22.5 度 (FanGraphs 用 15度)
    PULL_THRESHOLD = 15.0  
    
    # --- 步驟 3: 建立分類邏輯 ---
    # 使用 np.select 比較快 (比 apply 快很多)
    conditions = [
        # 情況 A: 右打者 (R)
        (df['stand'] == 'R') & (df['spray_angle'] < -PULL_THRESHOLD), # 拉打 (左邊)
        (df['stand'] == 'R') & (df['spray_angle'] > PULL_THRESHOLD),  # 推打 (右邊)
        
        # 情況 B: 左打者 (L)
        (df['stand'] == 'L') & (df['spray_angle'] > PULL_THRESHOLD),  # 拉打 (右邊)
        (df['stand'] == 'L') & (df['spray_angle'] < -PULL_THRESHOLD), # 推打 (左邊)
    ]
    
    choices = ['Pull', 'Oppo', 'Pull', 'Oppo']
    
    # 如果都不符合上述條件，那就是 Center (Straight)
    df['batted_ball_direction'] = np.select(conditions, choices, default='Center')
    
    return df

def analyze_spray_distribution(df):
    # 1. 確保有 spray_angle (沿用之前的公式)
    # 如果還沒算，記得先呼叫 calculate_spray_angle(df)
    
    # 2. 定義拉打/推打門檻 (通常用 15度)
    PULL_THRESHOLD = 15.0
    
    def get_quality(row):
        angle = row['spray_angle']
        stand = row['stand']
        
        # 邏輯判斷
        if stand == 'L':
            if angle > PULL_THRESHOLD: return 'Pull (Right)'  # 左打拉到右邊
            elif angle < -PULL_THRESHOLD: return 'Oppo (Left)' # 左打推到左邊
            else: return 'Center'
        else: # stand == 'R'
            if angle < -PULL_THRESHOLD: return 'Pull (Left)'  # 右打拉到左邊
            elif angle > PULL_THRESHOLD: return 'Oppo (Right)' # 右打推到右邊
            else: return 'Center'

    # 3. 應用分類
    df['contact_type'] = df.apply(get_quality, axis=1)
    
    # 4. 組合 "左右打 + 擊球型態"
    df['strategy_class'] = df['stand'] + "-" + df['contact_type']
    
    return df

def classify_pull_oppo(df):
    """
    輸入: 包含 'hc_x', 'hc_y', 'stand' 的 DataFrame
    輸出: 增加 'spray_angle' 和 'batted_ball_direction' 欄位
    """
    
    # --- 步驟 1: 先算出噴灑角度 (Spray Angle) ---
    # 0度=中外野, 負值=左外野, 正值=右外野
    hc_x_origin = 125.42
    hc_y_origin = 198.27
    df['location_x'] = (df['hc_x'] - hc_x_origin) * 2.495
    df['location_y'] = (hc_y_origin - df['hc_y']) * 2.495
    df['spray_angle'] = np.degrees(np.arctan2(df['location_x'], df['location_y']))
    
    # --- 步驟 2: 定義判斷標準 (Thresholds) ---
    # 業界標準通常定為 15 度 或 22.5 度 (FanGraphs 用 15度)
    PULL_THRESHOLD = 15.0  
    
    # --- 步驟 3: 建立分類邏輯 ---
    # 使用 np.select 比較快 (比 apply 快很多)
    conditions = [
        # 情況 A: 右打者 (R)
        (df['stand'] == 'R') & (df['spray_angle'] < -PULL_THRESHOLD), # 拉打 (左邊)
        (df['stand'] == 'R') & (df['spray_angle'] > PULL_THRESHOLD),  # 推打 (右邊)
        
        # 情況 B: 左打者 (L)
        (df['stand'] == 'L') & (df['spray_angle'] > PULL_THRESHOLD),  # 拉打 (右邊)
        (df['stand'] == 'L') & (df['spray_angle'] < -PULL_THRESHOLD), # 推打 (左邊)
    ]
    
    choices = ['Pull', 'Oppo', 'Pull', 'Oppo']
    
    # 如果都不符合上述條件，那就是 Center (Straight)
    df['batted_ball_direction'] = np.select(conditions, choices, default='Center')
    
    return df

def analyze_spray_distribution(df):
    # 1. 確保有 spray_angle (沿用之前的公式)
    # 如果還沒算，記得先呼叫 calculate_spray_angle(df)
    
    # 2. 定義拉打/推打門檻 (通常用 15度)
    PULL_THRESHOLD = 15.0
    
    def get_quality(row):
        angle = row['spray_angle']
        stand = row['stand']
        
        # 邏輯判斷
        if stand == 'L':
            if angle > PULL_THRESHOLD: return 'Pull (Right)'  # 左打拉到右邊
            elif angle < -PULL_THRESHOLD: return 'Oppo (Left)' # 左打推到左邊
            else: return 'Center'
        else: # stand == 'R'
            if angle < -PULL_THRESHOLD: return 'Pull (Left)'  # 右打拉到左邊
            elif angle > PULL_THRESHOLD: return 'Oppo (Right)' # 右打推到右邊
            else: return 'Center'

    # 3. 應用分類
    df['contact_type'] = df.apply(get_quality, axis=1)
    
    # 4. 組合 "左右打 + 擊球型態"
    df['strategy_class'] = df['stand'] + "-" + df['contact_type']
    
    return df


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