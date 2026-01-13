import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
from typing import Dict
import base64

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


class Config:
    def __init__(
        self,
        weights: Dict[str, float] = None,
        data_dir: str = '/neodata/open_dataset/mlb_data/preprocessed',
        filename: str = 'rtheta_prob_tbl.parquet'
    ):
        if weights is None:
            self.weights = {
                'single': 1.0,
                'double': 2.0,
                'triple': 3.0,
                'home_run': 4.0
            }
        else:
            self.weights = weights
            
        self.data_dir = data_dir
        self.filename = filename
        self.file_path = os.path.join(self.data_dir, self.filename)
        

def get_expected_bases_map(config: Config):
                           
    if not os.path.exists(config.file_path):
        raise FileNotFoundError(f"Probability table file not found: {config.file_path}")
    
    prob_df = pd.read_parquet(config.file_path)
    prob_pivot = prob_df.pivot(index='r_theta', columns='events', values='probability').fillna(0)
    
    prob_pivot['expected_bases'] = 0.0
    for event, w in config.weights.items():
        if event in prob_pivot.columns:
            prob_pivot['expected_bases'] += prob_pivot[event] * w
            
    return prob_pivot['expected_bases']

def prepare_regression_data(df: pd.DataFrame, 
                            exp_map: pd.Series,
                            config: Config):
    df_bip = df[df['description'] == 'hit_into_play'].copy()
    df_bip['expected_metric'] = df_bip['r_theta'].map(exp_map).fillna(0)
    event_weights = config.weights
    df_bip['real_metric'] = df_bip['events'].map(event_weights).fillna(0)

    group_cols = ['game_year', 'home_team', 'pitcher_team', 'batter_team']
    
    if 'game_pk' in df_bip.columns:
        group_cols.append('game_pk')
    else:
        group_cols.append('game_date') 

    agg_df = df_bip.groupby(group_cols).agg({
        'real_metric': 'sum',
        'expected_metric': 'sum',
        'events': 'count' 
    }).reset_index()
    
    agg_df.rename(columns={'events': 'weight', 'real_metric': 'sum_real', 'expected_metric': 'sum_exp'}, inplace=True)
    
    agg_df = agg_df[agg_df['sum_exp'] > 1.0].copy()
    
    agg_df['log_ratio'] = np.log((agg_df['sum_real'] + 0.5) / (agg_df['sum_exp'] + 0.5))
    agg_df['response'] = agg_df['sum_real'] - agg_df['sum_exp']
    
    agg_df['park'] = agg_df['home_team']
    agg_df['defense'] = agg_df['pitcher_team']
    
    return agg_df


def run_year_regression(data, year):
    
    data_yr = data[data['game_year'] == year].copy()

    # Fit WLS
    # Model: Y = Beta0 + Park + Defense
    # Statsmodels drops one category (reference) for Park and Defense
    model = smf.wls("response ~ C(park) + C(defense)", data=data_yr, weights=data_yr['weight'])
    #res = model.fit()
    res = model.fit_regularized(method='elastic_net', L1_wt=0.1, alpha=0.05)
    
    params = res.params
    
    all_parks = sorted(data_yr['park'].unique())
    all_defenses = sorted(data_yr['defense'].unique())
    
    beta1 = {} 
    beta2 = {} 
    
    beta0_raw = params['Intercept']
    
    # Fill Beta1 (Park)
    # Statsmodels naming: C(park)[T.TeamName]
    for p in all_parks:
        key = f"C(park)[T.{p}]"
        if key in params:
            beta1[p] = params[key]
        else:
            beta1[p] = 0.0
            
    # Fill Beta2 (Defense)
    for d in all_defenses:
        key = f"C(defense)[T.{d}]"
        if key in params:
            beta2[d] = params[key]
        else:
            beta2[d] = 0.0
            
    mean_beta1 = np.mean(list(beta1.values()))
    mean_beta2 = np.mean(list(beta2.values()))
    
    beta1_adj = {k: v - mean_beta1 for k, v in beta1.items()}
    beta2_adj = {k: v - mean_beta2 for k, v in beta2.items()}
    beta0_adj = beta0_raw + mean_beta1 + mean_beta2
    park_factors = {k: 100 * np.exp(v) for k, v in beta1_adj.items()}
    defense_factors = {k: 100 * np.exp(v) for k, v in beta2_adj.items()}
    
    return {
        'year': year,
        'intercept': beta0_adj,
        'park_factors': park_factors,
        'defense_factors': defense_factors
    }

def get_local_image_b64(logos_dir, team_name):
    file_path = os.path.join(logos_dir, f"{team_name}.png")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8').replace("\n", "").replace("\r", "")
    return f"data:image/png;base64,{encoded}"
