#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

from pybaseball import playerid_reverse_lookup, batting_stats, pitching_stats
from calculate_score import combined_score_tbl, batter_data_fg, pitcher_data_fg
from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl

df = get_truncated_dataset_with_team()
df = df[df['game_type'] == 'R']

# 2014 沒有 la and ls 
#print(df[df['game_year'] == 2015].sample(10))

yr = 2015
nyy_batter_df = df[
    (df['game_year'] == yr)&
    (df['batter_team'] == 'NYY')
]

def generate_team_events_tbl(
        team: str,
        data: pd.DataFrame,
        dist_df: pd.DataFrame,
        year: int,
        player_type: str,
        method: str = 'expectation',
        pitcher_data: pd.DataFrame = pitcher_data_fg,
        batter_data: pd.DataFrame = batter_data_fg
):
    """
    根據球隊與年份，計算整隊的事件統計表（例如 single、double、walk 等加總結果）
    """

    # 篩選該隊、該年度的資料
    if player_type == 'pitcher':
        team_year_df = data[(data['game_year'] == year) &
                            (data['pitcher_team'] == team)]
    elif player_type == 'batter':
        team_year_df = data[(data['game_year'] == year) &
                            (data['batter_team'] == team)]
    else:
        raise ValueError("player_type 必須是 'pitcher' 或 'batter'")
    
    # 抓出所有球員的 MLB ID
    player_ids = (
        team_year_df['pitcher'].unique().tolist()
        if player_type == 'pitcher'
        else team_year_df['batter'].unique().tolist()
    )

    team_tbls = []
    for pid in player_ids:
        try:
            score_tbl = combined_score_tbl(
                data=team_year_df[team_year_df[player_type] == pid],
                dist_df=dist_df,
                year=year,
                player_mlbid=pid,
                player_type=player_type,
                method=method,
                pitcher_data=pitcher_data,
                batter_data=batter_data
            )
            score_tbl[f"{player_type}_id"] = pid
            team_tbls.append(score_tbl)
        except Exception as e:
            print(f"⚠️ {team} {player_type} {pid} 發生錯誤：{e}")

    # 合併並加總
    team_tbls = pd.concat(team_tbls, ignore_index=True)
    team_score_tbl = (
        team_tbls.groupby("events", as_index=False)[["sum_real_count", "sum_expected_count"]].sum()
    )

    print(f"✅ 完成 {year} {team} {player_type} team 統計表")
    return team_score_tbl


nyy_14_batting_tbl = generate_team_events_tbl(
    team = 'NYY',
    data = get_truncated_dataset_with_team(),
    dist_df = get_rtheta_prob_tbl(),
    year = 2015,
    player_type = 'pitcher',
    method = 'expectation',
    pitcher_data = pitcher_data_fg,
    batter_data = batter_data_fg
)

print(nyy_14_batting_tbl)

#%%