#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from typing import Dict
from IPython.display import display as dp

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl



def get_expected_value_map(weights: Dict[str, float]) -> pd.Series:
    """
    根據傳入的權重字典，計算 r_theta 的預期價值。
    
    Args:
        weights: 事件權重字典。
                 例如: {'single': 1, 'home_run': 4}
    """
    # 1. 取得機率表
    prob_df = get_rtheta_prob_tbl()
    
    # 2. 轉置
    prob_pivot = prob_df.pivot(index='r_theta', columns='events', values='probability').fillna(0)
    
    # 3. 計算
    prob_pivot['expected_value'] = 0.0
    
    for event, w in weights.items():
        if event in prob_pivot.columns:
            prob_pivot['expected_value'] += prob_pivot[event] * w
        else:
            # 提醒使用者傳了沒用的 key
            print(f"Warning: there is no '{event}' in the data, ignore this weight.")
            
    return prob_pivot['expected_value']


# slg_tbl = get_expected_value_map(weights_slg)
# avg_tbl = get_expected_value_map(weights_avg)

# combined_tbl = pd.concat([slg_tbl, avg_tbl], axis=1, 
# keys=['Expected_SLG', 'Expected_AVG'])

# dp(combined_tbl.sample(10))

def prepare_regression_data(df, 
                            exp_map: pd.Series, 
                            weights: Dict[str, float]):
    """
    Prepare data for weighted regression:
    - Filter for Hit Into Play
    - Calculate Real Total Bases and Expected Total Bases per play
    - Aggregate by Game-Team (unique matchup of Park, Defense, Offense)
    - Calculate Y = log(Real / Expected)
    - Calculate Weight = Count of BIP
    """
    # Filter for hits into play
    # Using description 'hit_into_play' is safer than mapping events
    df_bip = df[df['description'] == 'hit_into_play'].copy()

    # Map expected total bases
    df_bip['expected_tb'] = df_bip['r_theta'].map(exp_map).fillna(0)
    # Calculate Real Metric (Total Bases for SLG)
    event_weights = weights
    df_bip['real_tb'] = df_bip['events'].map(event_weights).fillna(0)


    # Aggregate by Game-Team
    # We group by game_year, home_team (Park), pitcher_team (Defense), batter_team
    # We use game_pk if available, else use date/teams as proxy. 
    # The truncated dataset seems not to have game_pk in previous generic inspection, 
    # but grouping by (game_date, home_team, batter_team) is sufficient to identify a game-half.
    # Also include game_year for splitting later.
    
    # Check if 'game_pk' exists
    group_cols = ['game_year', 'home_team', 'pitcher_team', 'batter_team']
    if 'game_pk' in df_bip.columns:
        group_cols.append('game_pk')
    else:
        group_cols.append('game_date') # Proxy for unique game ID

    agg_df = df_bip.groupby(group_cols).agg({
        'real_tb': 'sum',
        'expected_tb': 'sum',
        'events': 'count' # Weight
    }).reset_index()
    
    #### 把 tb 改成 count
    agg_df.rename(columns={'events': 'weight', 'real_tb': 'sum_real_count', 'expected_tb': 'sum_exp_count'}, 
                    inplace=True)
    
    # Filter out empty weights or negligible expected values
    # If sum_exp is 0, we cannot estimate a factor.
    # Expected bases for a full game should be >> 1.
    agg_df = agg_df[agg_df['sum_exp_count'] > 1.0].copy()
    
    agg_df['log_ratio'] = agg_df['sum_real_count'] - agg_df['sum_exp_count']
    
    # Define Park and Defense columns explicitly for formula
    agg_df['park'] = agg_df['home_team']
    agg_df['defense'] = agg_df['pitcher_team']
    
    return agg_df


def run_year_regression(data, year):
    """
    Run WLS for a specific year and return adjusted coefficients.
    """
    # Filter by year
    data_yr = data[data['game_year'] == year].copy()
    
    if len(data_yr) < 100:
        print(f"Skipping {year}: Not enough data ({len(data_yr)} rows)")
        return None

    # Fit WLS
    # Model: Y = Beta0 + Park + Defense
    # Statsmodels automatically drops one category (reference) for Park and Defense
    mod = smf.wls("log_ratio ~ C(park) + C(defense)", data=data_yr, weights=data_yr['weight'])
    res = mod.fit()
    
    # Extract and Adjust Coefficients
    params = res.params
    
    all_parks = sorted(data_yr['park'].unique())
    all_defenses = sorted(data_yr['defense'].unique())
    
    # Reconstruct Full Dictionaries
    beta1 = {} # Park Coefficients
    beta2 = {} # Defense Coefficients
    
    beta0_raw = params['Intercept']
    
    # Fill Beta1 (Park)
    # Statsmodels naming: C(park)[T.TeamName]
    for p in all_parks:
        # Check if this park is the reference (omitted) or present
        key = f"C(park)[T.{p}]"
        if key in params:
            beta1[p] = params[key]
        else:
            # This is the Reference Category
            beta1[p] = 0.0
            
    # Fill Beta2 (Defense)
    for d in all_defenses:
        key = f"C(defense)[T.{d}]"
        if key in params:
            beta2[d] = params[key]
        else:
            # Reference
            beta2[d] = 0.0
            
    # --- Adjustment Step (as requested) ---
    # 1. Calculate Average of betas
    mean_beta1 = np.mean(list(beta1.values()))
    mean_beta2 = np.mean(list(beta2.values()))
    
    # 2. Update coeffs: beta_new = beta_old - mean
    beta1_adj = {k: v - mean_beta1 for k, v in beta1.items()}
    beta2_adj = {k: v - mean_beta2 for k, v in beta2.items()}
    
    # 3. Update Intercept: beta0_new = beta0_old + mean1 + mean2
    beta0_adj = beta0_raw + mean_beta1 + mean_beta2
    
    # --- Convert to Factors (100 * exp) ---
    # Park Factor: > 100 means Hitter Friendly (Real > Exp)
    park_factors = {k: 100 * np.exp(v) for k, v in beta1_adj.items()}
    
    # Defense Factor: > 100 means Bad Defense (Real > Exp, i.e., allows more hits)
    # Should check if user wants "Defense Strength" (where > 100 is GOOD).
    # Usually "Factor" on offensive metric implies >100 is "More Offense".
    # So >100 Defense Factor -> More Offense allowed -> Bad Defense.
    # If user wants "Defense Value", might need to invert. 
    # But sticking to "Factor" definition: 100*exp(beta).
    defense_factors = {k: 100 * np.exp(v) for k, v in beta2_adj.items()}
    
    return {
        'year': year,
        'intercept': beta0_adj,
        'park_factors': park_factors,
        'defense_factors': defense_factors
    }

if __name__ == "__main__":

    # 1. dict of weights
    WEIGHT_dict = {
        "avg": {'single': 1, 'double': 1, 'triple': 1, 'home_run': 1},
        "slg": {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4},
        # "woba": {'walk': 0.69, 'single': 0.88, 'double': 1.26, 'triple': 1.6, 'home_run': 2.07}
    }

    # 2. select weights
    index = "avg"  
    chosen_weights = WEIGHT_dict[index]

    # 3. loading data
    df = get_truncated_dataset_with_team()
    expected_value_map = get_expected_value_map(weights=chosen_weights)
    
    # 4. prepare regression data
    print("Preparing regression data...")
    reg_df = prepare_regression_data(df = df, 
                                    exp_map = expected_value_map, 
                                    weights = chosen_weights)
    
    # 5. run regression
    years = sorted(reg_df['game_year'].unique())
    results = []
    
    print(f"Running regressions for years: {years}")
    
    for yr in years:
        print(f"--- Processing {yr} ---")
        res = run_year_regression(reg_df, yr)
        if res:
            results.append(res)
            
    # 6. save results to csv
    # Create rows: Year, Team, ParkFactor, DefenseFactor
    output_rows = []
    for r in results:
        yr = r['year']
        # Union of teams in park and defense (should be same 30 teams)
        teams = set(r['park_factors'].keys()) | set(r['defense_factors'].keys())
        for tm in teams:
            output_rows.append({
                'Year': yr,
                'Team': tm,
                'ParkFactor': r['park_factors'].get(tm, np.nan),
                'DefenseFactor': r['defense_factors'].get(tm, np.nan),
                'Intercept': r['intercept']
            })
            
    out_df = pd.DataFrame(output_rows)
    save_path = f"/Users/yantianli/factor-and-defense-factor/estimated_factors_via_{index}.csv"
    out_df.to_csv(save_path, index=False)
    print(f"Successfully saved estimated factors to {save_path}")
    
    # 6. display snippet
    dp(out_df.head())
#%%