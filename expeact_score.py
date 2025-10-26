#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from create_frame import savant_data_14_24

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示


savant_data_14_24 = pd.read_parquet("/Users/yantianli/factor-and-defense-factor/savant_data_14_24_with_rtheta.parquet")


event_distribution = savant_data_14_24.groupby('r_theta')['events'].value_counts(normalize=True).reset_index()
event_distribution.columns = ['r_theta', 'events', 'probability']


def assign_expected_events(df, dist_df):
    """
    df: 原始資料，需包含 'r_theta'
    dist_df: 機率表，欄位 ['r_theta', 'events', 'probability']
    回傳: 原始 df 多一欄 'expected_events'
    """

    # 建立分佈字典
    dist_dict = {}
    for rtheta_value, sub in dist_df.groupby('r_theta'):
        events = sub['events'].values
        probs = sub['probability'].values
        dist_dict[rtheta_value] = (events, probs)

    def reassign(group):
        """
        對同一個 r_theta 的 group, 從 dist_dict 裡取出機率,
        用 np.random.choice() 為該 group 內每一筆資料抽樣事件.
        """
        rtheta_val = group['r_theta'].iloc[0]
        if rtheta_val in dist_dict:
            events, probs = dist_dict[rtheta_val]
            # 若機率和不精準，可再正規化一次
            probs = probs / probs.sum()
            chosen = np.random.choice(events, size=len(group), p=probs)
            group = group.copy()
            group['expected_events'] = chosen
        else:
            group = group.copy()
            group['expected_events'] = np.nan
        return group

    # 以 r_theta 分組後重抽
    df_out = df.groupby('r_theta', group_keys=False).apply(reassign)
    return df_out

df = assign_expected_events(savant_data_14_24, event_distribution)


cols = ['pitch_type', 'game_date', 'pitcher', 'batter', 'events', 'expected_events', 'launch_speed', 'launch_angle', 'r_theta']

expected_event_df = df[cols]

def get_expected_dataset():
    """
    提供給其他專案匯入使用。
    回傳：包含 expected_events 的 DataFrame。
    """
    path = "/Users/yantianli/factor-and-defense-factor/savant_data_14_24_with_expected_selected.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}\n請先執行 expeact_score.py 產生它。")
    return pd.read_parquet(path)

if __name__ == "__main__":
    # 若是直接執行這支檔案，就自動產生結果並輸出
    print("正在建立 expected_events 的資料集...")
    save_path = "/Users/yantianli/factor-and-defense-factor/savant_data_14_24_with_expected_selected.parquet"
    expected_event_df.to_parquet(save_path)
    print(f"✅ 已輸出結果到：{save_path}")

#%%