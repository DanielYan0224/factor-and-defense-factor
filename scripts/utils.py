import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf

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


def get_expected_bases_map(data_dir: str = '/neodata/open_dataset/mlb_data/preprocessed',
                           filename: str = 'rtheta_prob_tbl.parquet'):
    """
    Generate a mapping from r_theta bin to Expected Value based on metric.
    data_dir: Directory where the probability table is stored.
    filename: Name of the parquet file containing r_theta probability table.
    
    Returns a Pandas Series mapping r_theta to expected bases (SLG).
    """
    
    file_path = os.path.join(data_dir, filename)
    prob_df = pd.read_parquet(file_path)
    prob_pivot = prob_df.pivot(index='r_theta', columns='events', values='probability').fillna(0)
    weights = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    
    prob_pivot['expected_bases'] = 0.0
    for event, w in weights.items():
        if event in prob_pivot.columns:
            prob_pivot['expected_bases'] += prob_pivot[event] * w
            
    return prob_pivot['expected_bases']

def prepare_regression_data(df, exp_map):
    df_bip = df[df['description'] == 'hit_into_play'].copy()
    df_bip['expected_metric'] = df_bip['r_theta'].map(exp_map).fillna(0)
    event_weights = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
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
    
    agg_df['park'] = agg_df['home_team']
    agg_df['defense'] = agg_df['pitcher_team']
    
    return agg_df


def run_year_regression(data, year):
    
    data_yr = data[data['game_year'] == year].copy()

    # Fit WLS
    # Model: Y = Beta0 + Park + Defense
    # Statsmodels drops one category (reference) for Park and Defense
    model = smf.wls("log_ratio ~ C(park) + C(defense)", data=data_yr, weights=data_yr['weight'])
    res = model.fit()
    
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