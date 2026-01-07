#%%

import pandas as pd
import numpy as np
import os
from utils import get_expected_bases_map, prepare_regression_data, run_year_regression
import argparse

#%%

def main():
    parser = argparse.ArgumentParser(description="Run Weighted Regression on MLB Data")
    parser.add_argument('--data_dir', type=str, default='/neodata/open_dataset/mlb_data/preprocessed',
                        help='Directory containing preprocessed MLB data')
    parser.add_argument('--input_filename', type=str, default='truncated_data_with_rtheta_team.parquet',
                        help='Filename for the truncated dataset with r_theta and team info')
    parser.add_argument('--prob_table', type=str, default='rtheta_prob_tbl.parquet',
                        help='Filename for the r_theta probability table')
    parser.add_argument('--output_dir', type=str, default='/neodata/open_dataset/mlb_data/results',
                        help='Directory to save regression results')
    parser.add_argument('--output_filename', type=str, default='estimated_factors.csv',
                        help='Filename to save regression results')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    input_filename = args.input_filename
    prob_table_filename = args.prob_table

    exp_map = get_expected_bases_map(data_dir=data_dir, filename=prob_table_filename)
    truncated_file_path = os.path.join(data_dir, input_filename)
    df = pd.read_parquet(truncated_file_path)
    reg_df = prepare_regression_data(df, exp_map)

    years = sorted(reg_df['game_year'].unique())
    results = []

    for year in years:
        result = run_year_regression(reg_df, year)
        results.append(result)

    output = []
    for result in results:
        year = result['year']
        teams = set(result['park_factors'].keys()) | set(result['defense_factors'].keys())
        for team in teams:
            output.append({
                'Year': year,
                'Team': team,
                'ParkFactor': result['park_factors'].get(team, np.nan),
                'DefenseFactor': result['defense_factors'].get(team, np.nan),
                'Intercept': result['intercept']
            })
    out_df = pd.DataFrame(output)
    output_path = os.path.join(args.output_dir, args.output_filename)
    out_df.to_csv(output_path, index=False)
    print(f"Successfully saved estimated factors to {output_path}")

if __name__ == "__main__":
    main()

#%%

# data_dir = '/neodata/open_dataset/mlb_data/preprocessed'
# prob_table_filename = 'rtheta_prob_tbl.parquet'
# exp_map = get_expected_bases_map(data_dir=data_dir, filename=prob_table_filename)

# truncated_filename = 'truncated_data_with_rtheta_team.parquet'
# truncated_file_path = os.path.join(data_dir, truncated_filename)
# df = pd.read_parquet(truncated_file_path)
# reg_df = prepare_regression_data(df, exp_map)

# year = 2024
# results = run_year_regression(reg_df, year)






#%%






#%%