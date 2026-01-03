import pandas as pd
import numpy as np


def add_rtheta_features(df):
   
    mask = (df["description"] == "hit_into_play") & (df["game_type"] == "R")
    work_df = df.loc[mask, ['launch_speed', 'launch_angle']].copy()
    work_df = work_df.dropna(subset=['launch_angle', 'launch_speed'], how='any')
    work_df["launch_speed"] = work_df["launch_speed"].clip(upper=120)

    speed_bins = np.arange(0, 121, 3)
    angle_bins = np.arange(-90, 91, 3)

    work_df["r_bin"] = pd.cut(work_df["launch_speed"], bins=speed_bins, 
                              labels=False, include_lowest=True)
    work_df["theta_bin"] = pd.cut(work_df["launch_angle"], bins=angle_bins, 
                                  labels=False, include_lowest=True)

    work_df["r_theta"] = (
        "r" + work_df["r_bin"].astype(int).astype(str) + 
        "_t" + work_df["theta_bin"].astype(int).astype(str)
    )

    df["r_bin"] = np.nan
    df["theta_bin"] = np.nan
    df["r_theta"] = None 

    df.loc[work_df.index, ["r_bin", "theta_bin", "r_theta"]] = \
        work_df[["r_bin", "theta_bin", "r_theta"]]

    return df


def assign_pitcher_batter_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    is_top = df['inning_topbot'] == 'Top'
    df['pitcher_team'] = np.where(is_top, df['home_team'], df['away_team'])
    df['batter_team'] = np.where(is_top, df['away_team'], df['home_team'])
    
    cols = list(df.columns)
    if 'away_team' in cols:
        insert_pos = cols.index('away_team') + 1
        for col in ['pitcher_team', 'batter_team']:
            if col in cols:
                cols.remove(col)
            cols.insert(insert_pos, col)
            insert_pos += 1
    
    return df[cols]     