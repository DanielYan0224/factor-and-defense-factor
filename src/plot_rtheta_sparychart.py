#%%
import pandas as pd
import os
import joblib

from IPython.display import display as dp
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from pathlib import Path

speed_bins = np.arange(0, 121, 3)
angle_bins = np.arange(-90, 91, 3)

# 設定路徑 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

input_path = DATA_PROCESSED / "truncated_data_with_rtheta.parquet"
df = pd.read_parquet(input_path)


unique_events = df["events"].dropna().unique()
colors = plt.cm.get_cmap("hsv", len(unique_events))
event_to_color = {ev: colors(i) for i, ev in enumerate(unique_events)}
event_to_color["unknown"] = (0.5, 0.5, 0.5, 0.5)  # 為 NaN 預留顏色
event_colors = df["events"].fillna("unknown").map(event_to_color)
event_colors = mcolors.to_rgba_array(event_colors.tolist())

# 移除 unknown 事件
valid_mask = df["events"].notna()
df = df[valid_mask]
event_colors = event_colors[valid_mask]

# 繪圖
angles = np.deg2rad(df["launch_angle"])
radii = df["launch_speed"]
area = (radii / radii.max()) * 20

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='polar')
sc = ax.scatter(angles, radii, c=event_colors, s=area, alpha=0.7)
ax.set_thetamin(-90) #type: ignore
ax.set_thetamax(90) #type: ignore

# 加上圖例
handles = [plt.Line2D([0], [0], marker='o', color='w', label=ev,
                markerfacecolor=event_to_color[ev], markersize=8)
            for ev in event_to_color.keys() if ev != "unknown"]
ax.legend(handles=handles, title='Event', loc='center left', bbox_to_anchor=(1.05, 0.5))

# 加上輔助圓弧線（例：r = 20, 40, 60, 80, 100, 120）
# 使用既有的 bins
for r in speed_bins:
    ax.plot(np.linspace(np.radians(-90), np.radians(90), 200),
            [r]*200, '-', color='gray', lw=0.5)

for theta in angle_bins:
    ax.plot(np.radians([theta]*200),
            np.linspace(0, 120, 200), '-', color='gray', lw=0.5)


plt.show()
#%%