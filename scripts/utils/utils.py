import pandas as pd
from pathlib import Path
from pybaseball import chadwick_register

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_REF = BASE_DIR / "data" / "reference"


def get_player_id_map():
    """
    取得球員 ID 對照表 (包含 MLBAM ID 和 Fangraphs ID)。
    如果本地沒有檔案，會嘗試下載並存檔。
    """
    map_path = DATA_RAW / "player_id_map.csv"
    
    if map_path.exists():
        # 如果檔案存在，直接讀取
        return pd.read_csv(map_path)
    
    print("⚠️ 未發現 ID 對照表，正在下載完整球員名單 (chadwick_register)...")
    try:
        # 下載完整的球員 ID 表 (只需要做一次)
        df_map = chadwick_register()
        
        # 建立目錄並存檔
        DATA_REF.mkdir(parents=True, exist_ok=True)
        df_map.to_csv(map_path, index=False)
        print(f"✅ ID 對照表已儲存至：{map_path}")
        return df_map
    except Exception as e:
        print(f"❌ 下載失敗：{e}")
        return pd.DataFrame()


def get_whole_dataset():
    """回傳完整的 Parquet 主資料集"""
    path = DATA_PROCESSED / "savant_data_14_24.parquet"
    if not path.exists():
        raise FileNotFoundError(f"找不到完整資料集：{path}")
    print(f"載入完整資料：{path}")
    return pd.read_parquet(path)