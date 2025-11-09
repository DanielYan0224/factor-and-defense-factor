#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl
from team_park_metrics import get_team_score
from league_score_tbl import get_league_tbl
from generate_matrix import collect_eqns

from sympy import symbols, Eq, linear_eq_to_matrix
import re

df = get_truncated_dataset_with_team().copy()
dist_df = get_rtheta_prob_tbl()

bat_df = get_team_score("bat")
pitch_df = get_team_score("pitch")
park_df = get_team_score("park")
league_summary_tbl = get_league_tbl()


batter_tm_col = df.pop('batter_team')
pitcher_tm_col = df.pop('pitcher_team')

new_batter_tm_col = df.columns.get_loc('batter') + 1 #type: ignore
new_pitcher_tm_col = df.columns.get_loc('pitcher') + 1 #type: ignore

df.insert(new_batter_tm_col, 'batter_team', batter_tm_col) #type: ignore
df.insert(new_pitcher_tm_col, 'pitcher_team', pitcher_tm_col) #type: ignore

from sympy import symbols, Eq, linear_eq_to_matrix

# 儲存所有年度的矩陣資料
all_years_matrix = {}

for year in range(2023, 2024):
    print(f"正在建立 {year} 年方程式矩陣...")

    # 收集三組方程式
    year_eqs = collect_eqns(
        data=df,
        park_data=park_df,
        pitch_data=pitch_df,
        batter_data=bat_df,
        league_tbl=league_summary_tbl,
        metric='SLG',
        yr=year
    )

    # 合併成單一 dict（park + home + away）
    merged_eqs = {}
    for group_name, eq_dict in year_eqs.items():
        merged_eqs.update(eq_dict)

    # ---- 把所有方程式轉成符號形式 ----
    equations = []
    symbols_set = set()

    for tm, eq_str in merged_eqs.items():
        # 移除空白再拆成左右兩邊
        lhs, rhs = eq_str.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()

        # 提取所有變數名稱（允許底線與數字）
        var_names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs)

        # 建立符號字典
        symbol_dict = {name: symbols(name) for name in var_names}

        # 把rhs的字串轉成 sympy expression
        expr = eval(rhs, symbol_dict)

        # lhs是常數 (y_value)
        y_val = float(lhs)
        equations.append(Eq(expr, y_val))

        # 收集符號
        symbols_set.update(expr.free_symbols)

    # 建立矩陣形式
    A, b = linear_eq_to_matrix(equations, list(symbols_set))
    all_years_matrix[year] = {"A": A, "b": b, "symbols": list(symbols_set)}

    print(f"完成 {year} 年矩陣生成，共 {len(equations)} 條方程式，變數數量：{len(symbols_set)}")
    print(f"\n📘 {year} 年方程式矩陣：")
    for i, eq in enumerate(equations):
        print(f"Eq{i+1}: {eq}")
    print("\nA x = b (矩陣形式)：")
    dp(pd.DataFrame(np.hstack([np.array(A).astype(float), np.array(b).astype(float)]),
                    columns=[*list(map(str, symbols_set)), 'b']))
# 之後你就能用 least squares 解
# 例如：
# x = np.linalg.lstsq(np.array(A).astype(float), np.array(b).astype(float), rcond=None)[0]


#%%