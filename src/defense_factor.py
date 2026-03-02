#%%
# Standard Library
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Data Science
import pandas as pd
import numpy as np

# Visualization
from IPython.display import display as dp

# Domain / Third Party
from pybaseball import team_fielding, team_ids, fielding_stats

# Local Modules
from expect_score import get_truncated_dataset_with_team
from utils import team_name_transfer_dict, transform_team_name, filter_defense_data

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

offical_df = pd.read_csv(DATA_PROCESSED / "defense_data_transfer_truncated.csv")

our_estimated_df = pd.read_csv(DATA_PROCESSED / "defense_factor_comparison.csv", header=[0,1])

def merge_defense_data(official_defense_df: pd.DataFrame, 
                       our_estimated_df: pd.DataFrame, 
                       metric: str = 'Def'):
    """
    official_long_df: 含有多個指標的長表格 ['FP', 'Def', 'DRS', 'DPS', 'UZR', 'OAA', 'UZR/150','RngR', 'Range']
    ours_raw_df: 你的估計係數表格 (可以是 Wide format 或 Long format)
    metric: 你想要比較的指標名稱 (預設為 'Def')
    """
    # --- 1. 定義安全的標準化函數 ---
    def safe_normalize_20(col):
        series = pd.to_numeric(col, errors='coerce')
        # 如果整欄都是 NaN (例如 2015 的 OAA)，直接回傳全 NaN
        if series.isnull().all() or series.std() == 0 or np.isnan(series.std()):
            return pd.Series(np.nan, index=series.index)
        return ((series - series.mean()) / series.std()) * 20 + 100

    # --- 2. 處理官方數據 ---
    # 萃取目標指標
    off_subset = official_defense_df[['Team', 'game_year', metric]].copy()
    off_df = off_subset.pivot(index='Team', columns='game_year', values=metric)
    
    # 執行標準化 (不直接轉 int，避免 NaN 報錯)
    off_norm_df = off_df.apply(safe_normalize_20, axis=0).round()

    # --- 3. 處理你的估計數據 ---
    # 注意：這裡的 iloc 截取邏輯必須確保 Team 是第一欄
    our_data = our_estimated_df.iloc[:, np.r_[0, 11:len(our_estimated_df.columns)]].copy()
    our_data.columns = np.r_[["Team"], [str(year) for year in range(2015, 2025)]]
    
    # 統一縮寫並設為 Index
    our_data['Team'] = our_data['Team'].str.strip().replace({"AZ": "ARI", "AZ ": "ARI"})
    our_data = our_data.set_index('Team')
    
    # 執行標準化
    ours_norm_df = our_data.apply(safe_normalize_20, axis=0).round()

    # --- 4. 合併表格 ---
    # 統一將欄位名轉為字串
    ours_norm_df.columns = [str(col) for col in ours_norm_df.columns]
    off_norm_df.columns = [str(col) for col in off_norm_df.columns]

    # 合併前先將 Team 轉為字串欄位，避免 dtype 衝突 (int64 vs object)
    left = ours_norm_df.reset_index()
    left['Team'] = left['Team'].astype(str)
    
    right = off_norm_df.reset_index()
    right['Team'] = right['Team'].astype(str)

    comparison_table = pd.merge(
        left, 
        right, 
        on='Team', 
        suffixes=('_Ours', '_Official')
    )

    # --- 5. 動態排序 (只排存在的欄位) ---
    years = [str(yr) for yr in range(2015, 2025)]
    ours_cols = [f"{yr}_Ours" for yr in years if f"{yr}_Ours" in comparison_table.columns]
    off_cols = [f"{yr}_Official" for yr in years if f"{yr}_Official" in comparison_table.columns]
    
    # 最終選擇欄位：只選取「非全空」的欄位
    final_order = ['Team'] + off_cols + ours_cols
    comparison_table = comparison_table[final_order]

    for col in comparison_table.columns:
        if col != 'Team':
            comparison_table[col] = pd.to_numeric(comparison_table[col], errors='coerce').astype('Int64')

    comparison_table = comparison_table.rename(columns={'Team': f'Team ({metric})'})

    return comparison_table


for def_metric in ['FP', 'Def', 'DRS', 'DPS', 'UZR', 'OAA', 'UZR/150', 'RngR', 'Range']:
    try:
        print(f"正在處理指標: {def_metric}...")
        
        # 執行合併
        defense_df = merge_defense_data(
            official_defense_df = offical_df, 
            our_estimated_df = our_estimated_df,
            metric = def_metric
        )
        
        # 儲存檔案
        # 建議確保 DATA_PROCESSED 路徑存在
        output_path = DATA_PROCESSED / f"defense_factor_{def_metric.replace('/', '_')}.csv"
        
        # index=False 是對的，因為 Team 已經在欄位裡了
        defense_df.to_csv(output_path, index=False)
        
        print(f"✅ {def_metric} 儲存成功！")
        
    except Exception as e:
        print(f"❌ {def_metric} 處理失敗，原因: {e}")

#%%
import matplotlib.pyplot as plt
from pathlib import Path

# Data Science
import pandas as pd
import numpy as np

# Visualization
from IPython.display import display as dp

# Domain / Third Party
from pybaseball import team_fielding, team_ids, fielding_stats

# Local Modules
from expect_score import get_truncated_dataset_with_team
from utils import team_name_transfer_dict, transform_team_name, filter_defense_data

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
PNG_DIR = BASE_DIR / "png"

# ['FP', 'Def', 'DRS', 'DPS', 'UZR', 'OAA', 'UZR/150', 'RngR', 'Range']
def_metric = 'UZR/150'
def_metric = def_metric.replace('/', '_')

defense_df = pd.read_csv(DATA_PROCESSED / f"defense_factor_{def_metric}.csv")


years = [str(yr) for yr in range(2015, 2025)]
teams = sorted(defense_df.iloc[:, 0].unique()) 

output_dir = PNG_DIR / f"{def_metric}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for team in teams:
    # 1. 提取該球隊的數據
    team_row = defense_df[defense_df.iloc[:, 0] == team].iloc[0]
    
    # 我們利用欄位名稱來精確抓取，這樣最保險
    years = [str(yr) for yr in range(2015, 2025)]
    
    ours_values = [team_row[f"{yr}_Ours"] for yr in years]
    off_values = [team_row[f"{yr}_Official"] for yr in years]
    
    # 轉成 float 才能繪圖 (Int64 會自動處理 NaN)
    ours_values = np.array(ours_values, dtype=float)
    off_values = np.array(off_values, dtype=float)

    # 3. 繪圖
    plt.figure(figsize=(10, 6))
    
    # 畫出我們的估計線 (實線)
    plt.plot(years, ours_values, marker='o', linestyle='-', linewidth=2, 
             color='#1f77b4', label='Our Estimated')
    
    # 畫出官方指標線 (虛線)
    plt.plot(years, off_values, marker='s', linestyle='--', linewidth=2, 
             color='#ff7f0e', label=f'Official {def_metric}')
    
    # 4. 裝飾圖表
    plt.axhline(100, color='black', linewidth=0.8, linestyle=':', alpha=0.5) # 100 基準線
    plt.title(f'{team} - {def_metric} Comparison (2015-2024)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Normalized Score (Mean=100)', fontsize=12)
    plt.ylim(40, 160) # 固定範圍讓 30 張圖具備可比性
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend()
    
    # 5. 儲存
    save_path = output_dir / f"{team}_{def_metric}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() # 務必關閉以釋放記憶體
    
    print(f"✅ 已儲存: {save_path}")

#%%