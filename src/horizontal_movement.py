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

# df_2015 = pd.read_csv(DATA_RAW / "statcast_2015.csv")
# df_2015_FC = df_2015[df_2015['pitch_type'] == 'FC'].copy()

# fig = px.box(
#     df_2015_FC,
#     x="pfx_z",
#     points=False,
#     title="SI Vertical Break (2015)",
#     labels={
#         "pfx_z": "|pfx_z| (inches)"
#     }
# )

# fig.show()
#%%
yrs = range(2015, 2025)
pitch_type = "ST"

df_list_r = []
df_list_l = []

for year in yrs:
    df_year = pd.read_csv(DATA_RAW / f"statcast_{year}.csv")
    df_year = df_year[
        df_year["pfx_x"].notna() & df_year["p_throws"].notna()
    ].copy()

    df_year = df_year[df_year["pitch_type"] == pitch_type].copy()
    df_year["year"] = str(year)
    df_year["pfx_x_in"] = df_year["pfx_x"] * 12

    df_r = df_year[df_year["p_throws"] == "R"].copy()
    df_l = df_year[df_year["p_throws"] == "L"].copy()

    df_list_r.append(df_r)
    df_list_l.append(df_l)

df_r_all = pd.concat(df_list_r, ignore_index=True)
df_l_all = pd.concat(df_list_l, ignore_index=True) 
#%%
df_all = pd.concat([df_r_all, df_l_all], ignore_index=True)

fig = px.box(
    df_all,
    x="year",
    y="pfx_x_in",
    color="p_throws",
    points=False,
    title=f"{pitch_type} Horizontal Break by Year (RHP vs LHP)",
    labels={
        "year": "Year",
        "pfx_x_in": "pfx_x (inches)",
        "p_throws": "Throws"
    }
)

fig.update_xaxes(
    type="category",
    categoryorder="array",
    categoryarray=[str(year) for year in yrs],
    tickmode="array",
    tickvals=[str(year) for year in yrs],
    ticktext=[str(year) for year in yrs]
)

fig.show()
#%%