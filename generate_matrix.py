#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl, get_whole_dataset
from team_park_metrics import get_team_score
from league_score_tbl import get_league_tbl


df = get_truncated_dataset_with_team().copy()
dist_df = get_rtheta_prob_tbl()

bat_df = get_team_score("bat")
pitch_df = get_team_score("pitch")
park_df = get_team_score("park")

league_summary_tbl = get_league_tbl()


batter_tm_col = df.pop('batter_team')
pitcher_tm_col = df.pop('pitcher_team')

new_batter_tm_col = df.columns.get_loc('batter') + 1 #type: ignore
new_pitcher_tm_col = df.columns.get_loc('pitcher') + 1 #type: ignore

df.insert(new_batter_tm_col, 'batter_team', batter_tm_col) #type: ignore
df.insert(new_pitcher_tm_col, 'pitcher_team', pitcher_tm_col) #type: ignore


def generate_home_park_defense(
            pitcher_data:pd.DataFrame,
            park_data:pd.DataFrame,
            league_tbl:pd.DataFrame,
            metric:str,
            yr:int):
    
    pitcher
#%%
def generate_park_equ(
            data:pd.DataFrame,
            park_data:pd.DataFrame,
            league_tbl:pd.DataFrame,
            metric:str,
            yr:int):
    # 只保留例行賽的 data
    df = data[data['game_type'] == 'R'].copy()
    tm_list = df['home_team']

    park_eqns = {}
    
    # ---- 計算聯盟整體校正值 ----
    league_correction = (
        league_tbl.loc[league_tbl['Year']==yr, f'ex_{metric}'].values[0] 
                        - league_tbl.loc[league_tbl['Year']==yr, f'real_{metric}'].values[0]
    )

    # ---- 對每個球場建立 equation ----
    for tm in tm_list.unique():
        # 篩出主場資料
        home_team_df = df[
            (df['game_year'] == yr) &
            (df['home_team'] == tm) &
            (df['description'] == 'hit_into_play')
        ]

        # 計算總打進場
        total_play = len(home_team_df)
        if total_play == 0:
            continue

        # 計算各隊在該主場的打進場比例
        away_play_count = (
            home_team_df.groupby('batter_team')['description']
            .count()
            .reset_index(name='play_count')
        )

        # 轉成比例並正規化
        away_play_count['ratio'] = away_play_count['play_count'] / away_play_count['play_count'].sum()

        # Round & Normalize
        away_play_count['ratio'] = away_play_count['ratio'].round(4)
        away_play_count['ratio'] /= away_play_count['ratio'].sum()

        # ---- 計算球隊校正值 ----
        park_row = park_data.loc[
            (park_data["Team"] == tm) & (park_data["Year"] == yr)
        ]
        park_correction = park_row[f'ex_{metric}'].values[0] - park_row[f'real_{metric}'].values[0]

        # ---- 計算最終 y 值 ----
        y_value = park_correction - league_correction

        # 建立方程式： y = base_term + summation coeff * defense_team
        base_term = f"park_factor_{tm}_{yr}"
        terms = [f"{row['ratio']:.4f} * defense_{row['batter_team']}_{yr}" 
                for _, row in away_play_count.iterrows()]
        year_factor = "year_factor"
        # 組合成完整方程
        eq = f"{y_value:.4f} = {year_factor} + {base_term} + " + " + ".join(terms)
        park_eqns[tm] = eq

    return park_eqns


park_eqs = generate_park_equ(
        data=get_truncated_dataset_with_team(),
        park_data=get_tm_park_score(),
        league_tbl=get_league_tbl(),
        metric="SLG",
        yr=2024)

def display_park_equations(eq_dict: dict):
    """以數學方程式的樣式印出所有球隊結果"""
    print("=== Park Factor Equations ===")
    for _, eq in eq_dict.items():
        dp(f"{eq}")








#display_park_equations(park_eqs)
#%%
# def generate_tm_park_defense(data:pd.DataFrame,
#                             data_2:pd.DataFrame,
#                             metric:str,
#                             yr:int):
#     """
#     計算在洋基球場的pf跟洋基的defense factor
#     """
#     data = 

# if __name__ == "__main__":
#     equations = generate_park_equ(data=get_truncated_dataset_with_team(),
#                                   park_data= get_tm_park_score(),
#                                   league_tbl= get_league_tbl(),
#                                   metric="SLG", 
#                                   yr=2024)
#     for team, eq in equations.items():
#         print(eq)



#%%