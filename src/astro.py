#%%
# Standard Library
import os
from pathlib import Path

# Data Science
import pandas as pd
import numpy as np
import joblib

# Visualization
from IPython.display import display as dp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors 
import plotly.express as px
#%%
pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

#%%
def event_count_tbl(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 PA-ending rows 依 events 彙總成 count table。
    回傳欄位至少包含:
    - events
    - sum_real_count
    """
    out = (
        df.groupby("events", dropna=False)
          .size()
          .reset_index(name="sum_real_count")
    )
    return out


def cal_pa_ab(df: pd.DataFrame, col: str = "sum_real_count"):
    """
    計算打席數 (PA) 與打數 (AB)。

    PA：所有 sum_real_count 的總和。
    AB：PA 扣除不計入打數的事件。
    """
    tmp = df.copy()

    all_events = [
        'strikeout', 'strikeout_double_play', 'walk', 'intent_walk', 'hit_by_pitch',
        'single', 'double', 'triple', 'home_run',
        'field_out', 'double_play', 'triple_play',
        'sac_fly', 'sac_bunt', 'catcher_interf',
        'sac_bunt_double_play', 'sac_fly_double_play'
    ]

    exclude_events = [
        'walk', 'intent_walk', 'sac_bunt', 'sac_fly',
        'catcher_interf', 'hit_by_pitch',
        'sac_bunt_double_play', 'sac_fly_double_play'
    ]

    tmp[col] = pd.to_numeric(tmp[col], errors='coerce').fillna(0)

    existing_events = set(tmp["events"].astype(str))
    missing_events = [evt for evt in all_events if evt not in existing_events]

    if missing_events:
        add_df = pd.DataFrame({
            "events": missing_events,
            col: 0
        })
        tmp = pd.concat([tmp, add_df], ignore_index=True)

    pa = tmp[col].sum()
    excluded_count = tmp.loc[tmp["events"].isin(exclude_events), col].sum()
    ab = pa - excluded_count

    return int(pa), int(ab)

def summarize_from_event_table(event_tbl: pd.DataFrame,
                               games: int,
                               runs: float,
                               col: str = "sum_real_count") -> dict:
    tmp = event_tbl.copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)

    def get_count(evt_list):
        return tmp.loc[tmp["events"].isin(evt_list), col].sum()

    pa, ab = cal_pa_ab(tmp, col=col)

    h = get_count(["single", "double", "triple", "home_run"])
    bb = get_count(["walk", "intent_walk"])
    hbp = get_count(["hit_by_pitch"])
    sf = get_count(["sac_fly", "sac_fly_double_play"])
    hr = get_count(["home_run"])

    singles = get_count(["single"])
    doubles = get_count(["double"])
    triples = get_count(["triple"])
    tb = singles + 2 * doubles + 3 * triples + 4 * hr

    avg = h / ab if ab > 0 else np.nan
    obp_den = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_den if obp_den > 0 else np.nan
    slg = tb / ab if ab > 0 else np.nan
    ops = obp + slg if pd.notna(obp) and pd.notna(slg) else np.nan
    hr_g = hr / games if games > 0 else np.nan
    r_g = runs / games if games > 0 else np.nan

    return {
        "G": int(games),
        "PA": int(pa),
        "AVG": round(avg, 3),
        "OBP": round(obp, 3),
        "SLG": round(slg, 3),
        "OPS": round(ops, 3),
        "HR": int(hr),
        "HR/G": round(hr_g, 3),
        "R/G": round(r_g, 3),
    }

#%%

df = pd.read_csv(DATA_RAW / "statcast_2015.csv")
df_2015 = df_2015[df_2015['game_type'] == 'R']

team_code = 'HOU'


pa_df = df_2015[df_2015["events"].notna()].copy()

pa_df["batting_team"] = np.where(
    pa_df["inning_topbot"] == "Top",
    pa_df["away_team"],
    pa_df["home_team"]
)

pa_df["runs_scored"] = (pa_df["post_bat_score"] - pa_df["bat_score"]).clip(lower=0)

hou_home_batting = pa_df[
    (pa_df["batting_team"] == team_code) &
    (pa_df["home_team"] == team_code)
].copy()

hou_other_batting = pa_df[
    (pa_df["batting_team"] == team_code) &
    (pa_df["away_team"] == team_code)
].copy()

opp_home_park_batting = pa_df[
    (pa_df["batting_team"] != team_code) &
    (pa_df["home_team"] == team_code)
].copy()

opp_other_park_batting = pa_df[
    (pa_df["batting_team"] != team_code) &
    (pa_df["away_team"] == team_code)
].copy()
#%%
all_at_hou_park = pa_df[
    pa_df["home_team"] == team_code
].copy()

league_avg_df = pa_df.copy()

table = pd.DataFrame.from_dict({
    "HOU (Home Park)": summarize_from_event_table(
        event_count_tbl(hou_home_batting),
        games=hou_home_batting["game_pk"].nunique(),
        runs=hou_home_batting["runs_scored"].sum()
    ),
    "HOU (Other ballpark)": summarize_from_event_table(
        event_count_tbl(hou_other_batting),
        games=hou_other_batting["game_pk"].nunique(),
        runs=hou_other_batting["runs_scored"].sum()
    ),
    "Opponents (Home Park)": summarize_from_event_table(
        event_count_tbl(opp_home_park_batting),
        games=opp_home_park_batting["game_pk"].nunique(),
        runs=opp_home_park_batting["runs_scored"].sum()
    ),
    "Opponents (Other ballpark)": summarize_from_event_table(
        event_count_tbl(opp_other_park_batting),
        games=opp_other_park_batting["game_pk"].nunique(),
        runs=opp_other_park_batting["runs_scored"].sum()
    ),
    "All teams at HOU Home Park": summarize_from_event_table(
        event_count_tbl(all_at_hou_park),
        games=all_at_hou_park["game_pk"].nunique(),
        runs=all_at_hou_park["runs_scored"].sum()
    ),
    "League Average": summarize_from_event_table(
        event_count_tbl(league_avg_df),
        games=league_avg_df["game_pk"].nunique(),
        runs=league_avg_df["runs_scored"].sum()
    )
}, orient="index")

dp(table)
#%%