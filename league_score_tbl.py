#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl, get_whole_dataset

from calculate_offensive_mertics import cal_pa_ab, cal_babip, cal_obp, cal_ops, cal_pa_ab, cal_slg

batter_data_fg = pd.read_csv("/Users/yantianli/factor_and_defense_factor/fg_batting.csv")
pitcher_data_fg = pd.read_csv("/Users/yantianli/factor_and_defense_factor/fg_pitching.csv")

def generate_league_hip_tbl(data: pd.DataFrame,
                          distribution: pd.DataFrame,
                          year: int):
    """
    建立年度整體 hit-into-play (HIP) 統計表。
    包含：
    - 各事件的真實次數 (real_df)
    - 根據 r_theta 機率分布計算的期望次數 (expected_df)
    - 並進行 normalization，讓期望總和與真實總和一致。
    """

    # 篩選該年度的打進場資料
    data = data[
        (data['game_year'] == year) &
        (data['game_type'] == 'R') &
        (data['description'] == 'hit_into_play')
    ].copy()
    data = data.dropna(subset=['events'])

    # ---- 計算真實事件分布 ----
    real_df = data['events'].value_counts().reset_index()
    real_df.columns = ['events', 'sum_real_count']

    # ---- 根據 r_theta 計算期望次數 ----
    counts_dict = data['r_theta'].value_counts().to_dict()
    year_tbl = distribution.copy()
    year_tbl["total_count"] = year_tbl["r_theta"].map(lambda x: counts_dict.get(x, 0))
    year_tbl["expected_count"] = year_tbl["probability"] * year_tbl["total_count"]

    # ---- normalize 讓期望總數 = 真實總數 ----
    real_event_count = real_df['sum_real_count'].sum()
    expected_total = year_tbl['expected_count'].sum()
    if expected_total > 0:
        year_tbl['expected_count'] *= (real_event_count / expected_total)

    year_tbl['expected_count'] = year_tbl['expected_count'].round(4)

    # ---- 聚合成事件層級 ----
    year_event_df = (
        year_tbl.groupby('events')['expected_count']
        .sum()
        .reset_index()
        .rename(columns={'expected_count': 'sum_expected_count'})
    )

    # ---- 合併真實與期望 ----
    year_event_df = pd.merge(year_event_df, real_df, on='events', how='outer').fillna(0)
    year_event_df = year_event_df.sort_values('sum_real_count', ascending=False).reset_index(drop=True)

    return year_event_df

def generate_league_nonhip_tbl(data: pd.DataFrame, 
                    year: int):
    """
    計算指定球員在指定年度的真實事件分布表。
    回傳：
        DataFrame，欄位 ['events', 'sum_real_count']
    """
    # 篩選出沒有打進場的 data 
    df = data[
        (data['game_year'] == year) & 
        (data['game_type'] == 'R') &
        (data['description'] != 'hit_into_play')
    ].copy()
    df = df.dropna(subset=['events'])

    # 統計每個事件的次數
    real_df = df['events'].value_counts().reset_index()
    real_df.columns = ['events', 'sum_real_count']
    real_df['sum_expected_count'] = real_df['sum_real_count']

    # 依照次數排序
    real_df = real_df.sort_values('sum_real_count', ascending=False).reset_index(drop=True)

    return real_df

def league_ibb_total(year: int, 
                    pitcher_data: pd.DataFrame = pitcher_data_fg,
                    batter_data: pd.DataFrame = batter_data_fg) -> int:
    """
    回傳該年度全聯盟的 IBB 總數（以打者資料為主）。
    """
    # 優先使用打者資料，因為 Fangraphs 的打者表裡有 IBB 欄
    df = batter_data
    if 'Intent_walk' not in df.columns:
        raise ValueError("Fangraphs 打者資料中找不到 IBB 欄位。")

    league_df = df[df['game_year'] == year]
    if league_df.empty:
        print(f"{year} 年的 Fangraphs 打者資料為空。")
        return 0

    return int(league_df['Intent_walk'].sum())

def generate_league_table(
    data: pd.DataFrame,
    distribution: pd.DataFrame,
    year: int,
    pitcher_data: pd.DataFrame = pitcher_data_fg,
    batter_data: pd.DataFrame = batter_data_fg
) -> pd.DataFrame:
    """
    建立全聯盟年度事件統計表（整合 HIP、non-HIP、IBB）。
    """
    # --- 各子表 ---
    league_hip_tbl = generate_league_hip_tbl(data, distribution, year)
    league_nonhip_tbl = generate_league_nonhip_tbl(data, year)
    league_ibb_total_val = league_ibb_total(year, pitcher_data, batter_data)

    # --- 將 IBB 轉成 DataFrame 格式 ---
    league_ibb_tbl = pd.DataFrame([{
        "events": "intent_walk",
        "sum_real_count": league_ibb_total_val,
        "sum_expected_count": league_ibb_total_val
    }])

    # --- 合併三個表 ---
    league_tbl = pd.concat(
        [league_hip_tbl, league_nonhip_tbl, league_ibb_tbl],
        ignore_index=True
    )

    # --- 確保排序一致性 ---
    league_tbl = league_tbl.sort_values("sum_real_count", ascending=False).reset_index(drop=True)

    return league_tbl

# league_24_tbl = generate_league_table(
#     data=get_truncated_dataset_with_team(),
#     distribution=get_rtheta_prob_tbl(),
#     year=2024,
#     pitcher_data=pitcher_data_fg,
#     batter_data=batter_data_fg)
# dp(league_24_tbl)

df = get_truncated_dataset_with_team()
dist_df = get_rtheta_prob_tbl()

league_summary = []
def generate_league_summary_tbl() -> pd.DataFrame:
    """
    產生 2015–2024 年（排除 2020）全聯盟年度統計表，
    並將結果輸出成 CSV。
    """
    df = get_truncated_dataset_with_team()
    dist_df = get_rtheta_prob_tbl()

    league_summary = []

    for year in [y for y in range(2015, 2025) if y != 2020]:
        year_league_score_tbl = generate_league_table(
            data=df,
            distribution=dist_df,
            year=year,
            pitcher_data=pitcher_data_fg,
            batter_data=batter_data_fg,
        )

        # --- 預期值 (Expected) ---
        ex_pa, ex_ab = cal_pa_ab(year_league_score_tbl, col='sum_expected_count')
        ex_ops = cal_ops(year_league_score_tbl, col='sum_expected_count')
        ex_slg = cal_slg(year_league_score_tbl, col='sum_expected_count')
        ex_obp = cal_obp(year_league_score_tbl, col='sum_expected_count')
        ex_babip = cal_babip(year_league_score_tbl, col='sum_expected_count')

        # --- 實際值 (Real) ---
        real_pa, real_ab = cal_pa_ab(year_league_score_tbl, col='sum_real_count')
        real_ops = cal_ops(year_league_score_tbl, col='sum_real_count')
        real_slg = cal_slg(year_league_score_tbl, col='sum_real_count')
        real_obp = cal_obp(year_league_score_tbl, col='sum_real_count')
        real_babip = cal_babip(year_league_score_tbl, col='sum_real_count')

        league_data = {
            "Year": year,
            # 預期值
            "ex_PA": ex_pa, "ex_AB": ex_ab,
            "ex_OPS": ex_ops, "ex_SLG": ex_slg, "ex_OBP": ex_obp, "ex_BABIP": ex_babip,
            # 實際值
            "real_PA": real_pa, "real_AB": real_ab,
            "real_OPS": real_ops, "real_SLG": real_slg, "real_OBP": real_obp, "real_BABIP": real_babip
        }

        league_summary.append(league_data)
        print(f"✅ 完成 {year} 年")

    league_df = pd.DataFrame(league_summary)
    save_path = "/Users/yantianli/factor_and_defense_factor/league_summary_tbl.csv"
    league_df.to_csv(save_path, index=False)
    print(f"📊 全聯盟統計表已儲存：{save_path}")
    return league_df

def get_league_tbl():
    """回傳全聯盟的打擊成績"""
    path = "/Users/yantianli/factor_and_defense_factor/league_summary_tbl.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到全聯盟資料：{path}")
    print(f"📦 載入聯盟打擊資料：{path}")
    return pd.read_csv(path)
#%%