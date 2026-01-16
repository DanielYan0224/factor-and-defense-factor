#%%
import pandas as pd
import numpy as np
from IPython.display import display as dp

from pybaseball import team_fielding, team_batting

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

defense_data = team_fielding(2023, 2024, ind=1)

def_col_mask = ['Def',                    # 綜合評價 (最重要)
                'DRS', 'UZR', 'OAA',]

defense_data = defense_data.loc[:, ['Season', 'Team'] + def_col_mask]
dp(defense_data.head(5))
#%%