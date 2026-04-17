import pandas as pd
import os
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# ─────────────────────────────────────────
# 1. LOAD & CLEAN (เหมือนเดิม)
# ─────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    if os.path.isfile(path) and path.endswith(".csv"):
        df = pd.read_csv(path)
        unnamed = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
    else:
        all_df = []
        for file in os.listdir(path):
            if file.endswith(".csv"):
                year = int(file.replace("gp_", "").replace(".csv", ""))
                tmp = pd.read_csv(os.path.join(path, file))
                tmp["year"] = year
                tmp.rename(columns={"pro_en": "province"}, inplace=True)
                tmp = tmp.drop(columns=["C"], errors="ignore")
                all_df.append(tmp)
        df = pd.concat(all_df, ignore_index=True)
    df["province"] = df["province"].str.strip()
    df = df.drop_duplicates()
    return df

# ─────────────────────────────────────────
# 1.1 LOAD OLD DATA FROM FOLDER (เพิ่มใหม่)
# ─────────────────────────────────────────
def load_old_data(start_year: int = None, end_year: int = None) -> pd.DataFrame:
    old_data_path = os.path.join("old_data", "old_data.csv")
    if not os.path.exists(old_data_path):
        return pd.DataFrame()
    df = pd.read_csv(old_data_path)
    
    # 🔧 แปลง year เป็นตัวเลข (int) เพื่อให้สามารถเปรียบเทียบได้
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    
    # กรองตามช่วงปี (ถ้าระบุ)
    if start_year is not None:
        df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]
    
    return df

# ─────────────────────────────────────────
# 1.2 SAVE CURRENT YEAR TO OLD_DATA (ปรับใหม่)
# ─────────────────────────────────────────
def save_current_to_old_data(df_current: pd.DataFrame):
    old_data_dir = "old_data"
    os.makedirs(old_data_dir, exist_ok=True)
    old_data_path = os.path.join(old_data_dir, "old_data.csv")
    
    if os.path.exists(old_data_path):
        df_old = pd.read_csv(old_data_path)
        # รวมข้อมูลเก่าและใหม่ (ต่อท้าย)
        df_combined = pd.concat([df_old, df_current], ignore_index=True)
        # ลบข้อมูลซ้ำ (ถ้ามีปีซ้ำกัน ให้เก็บปีล่าสุด? ตาม logic ปกติควรเก็บเฉพาะข้อมูลที่อัปเดต)
        # แต่เพื่อความปลอดภัย ให้ drop_duplicates โดยใช้ province และ year
        df_combined = df_combined.drop_duplicates(subset=["province", "year"], keep="last")
        df_combined.to_csv(old_data_path, index=False)
        print(f"Appended current year ({df_current['year'].iloc[0]}) to {old_data_path}")
    else:
        df_current.to_csv(old_data_path, index=False)
        print(f"Created new {old_data_path} with current year data")
# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING (เหมือนเดิม)
# ─────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"lag1", "lag2", "growth", "rolling_mean"}
    if required.issubset(df.columns):
        return df.sort_values(["province", "year"]).reset_index(drop=True)
    df = df.sort_values(["province", "year"]).reset_index(drop=True)
    grp = df.groupby("province")["CO2_tonnes"]
    df["lag1"] = grp.shift(1)
    df["lag2"] = grp.shift(2)
    df["growth"] = grp.pct_change()
    df["rolling_mean"] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    df = df.dropna(subset=['lag1', 'lag2']).reset_index(drop=True)
    return df

# ─────────────────────────────────────────
# 3. ENCODE & SCALE (เหมือนเดิม)
# ─────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame):
    # ถ้ายังมี NaN ใน lag1/lag2 ให้เติม median (เผื่อไว้)
    df["lag1"] = df.groupby("province")["lag1"].transform(lambda x: x.fillna(x.median()))
    df["lag2"] = df.groupby("province")["lag2"].transform(lambda x: x.fillna(x.median()))
    
    encoder = ce.BinaryEncoder(cols=["province"])
    df_enc = encoder.fit_transform(df)
    df_enc["growth"] = df_enc["growth"].fillna(0).clip(-1, 1)  
    
    num_cols = ["CO2_tonnes", "lag1", "lag2", "growth", "rolling_mean"]
    scaler = MinMaxScaler()
    df_enc[num_cols] = scaler.fit_transform(df_enc[num_cols])
    return df_enc, encoder, scaler

# ─────────────────────────────────────────
# 4. ANOMALY DETECTION (เหมือนเดิม)
# ─────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame, n_estimators: int = 200, contamination: float = 0.05) -> pd.DataFrame:
    features = ["CO2_tonnes", "lag1", "lag2", "growth", "rolling_mean"]
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    df = df.copy()
    df["anomaly"] = iso.fit_predict(df[features])
    print(f"Anomaly: {(df['anomaly'] == -1).sum()} | Normal: {(df['anomaly'] == 1).sum()}")
    return df

# ─────────────────────────────────────────
# 5. TRAIN & EVALUATE (เหมือนเดิม)
# ─────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame, scaler: MinMaxScaler, encoder: ce.BinaryEncoder, test_years: int = 3):
    target = "CO2_tonnes"
    drop_cols = [target, "anomaly"]
    unique_years = sorted(df["year"].unique())
    test_years_list = unique_years[-test_years:]
    train_years_list = unique_years[:-test_years]
    print(f"Train years: {train_years_list}")
    print(f"Test years: {test_years_list}")
    train = df[df["year"].isin(train_years_list)].copy()
    test = df[df["year"].isin(test_years_list)].copy()
    model_eval = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=5, random_state=42)
    model_eval.fit(train.drop(columns=drop_cols), train[target])
    preds_scaled = model_eval.predict(test.drop(columns=drop_cols))
    mape = mean_absolute_percentage_error(test[target], preds_scaled)
    r2 = r2_score(test[target], preds_scaled)
    print(f"Evaluation on {test_years} latest years - MAPE: {mape:.4f} | R²: {r2:.4f}")
    num_cols = [target, "lag1", "lag2", "growth", "rolling_mean"]
    test_real = test.copy()
    test_real[num_cols] = scaler.inverse_transform(test[num_cols])
    preds_inv_df = test[num_cols].copy()
    preds_inv_df[target] = preds_scaled
    preds_real = scaler.inverse_transform(preds_inv_df)[:, 0]
    test_decoded = encoder.inverse_transform(test_real.drop(columns=["anomaly", "preds"], errors="ignore"))
    test_decoded["preds"] = preds_real
    result = test_decoded[["province", "year", target, "preds"]].reset_index(drop=True)
    print("Training final model with all data...")
    model_final = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=5, random_state=42)
    model_final.fit(df.drop(columns=drop_cols), df[target])
    unique_years = sorted(df["year"].unique())
    print("Years in training data:", unique_years)
    return result, model_final ,mape,r2

# ─────────────────────────────────────────
# 6. PREDICT NEXT YEAR (ปรับให้รับ combined_data)
# ─────────────────────────────────────────
# เปลี่ยน predict_next_year ให้รับ n_years
def predict_next_year(model, encoder, scaler, combined_data, n_years=1) -> pd.DataFrame:
    all_preds = []
    current_data = combined_data.copy()

    for _ in range(n_years):
        max_year = current_data["year"].max()
        next_year = max_year + 1

        df_feat = add_features(current_data)
        rows = []
        for province, grp in df_feat.sort_values("year").groupby("province"):
            grp = grp.sort_values("year")
            co2_values = grp["CO2_tonnes"].values
            lag1 = co2_values[-1]
            lag2 = co2_values[-2] if len(co2_values) >= 2 else lag1
            growth = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            rolling_mean = co2_values[-3:].mean() if len(co2_values) >= 3 else co2_values.mean()
            rows.append({
                "province": province, "year": next_year, "CO2_tonnes": 0,
                "lag1": lag1, "lag2": lag2, "growth": growth, "rolling_mean": rolling_mean
            })

        df_next = pd.DataFrame(rows)
        df_enc = encoder.transform(df_next)
        df_enc["growth"] = df_enc["growth"].clip(-1, 1)
        num_cols = ["CO2_tonnes", "lag1", "lag2", "growth", "rolling_mean"]
        df_enc[num_cols] = scaler.transform(df_enc[num_cols])
        X_next = df_enc.drop(columns=["CO2_tonnes", "anomaly"], errors="ignore")
        preds_scaled = model.predict(X_next)

        preds_inv_df = df_enc[num_cols].copy()
        preds_inv_df["CO2_tonnes"] = preds_scaled
        preds_real = scaler.inverse_transform(preds_inv_df)[:, 0]

        result = df_next[["province", "year"]].copy()
        result["preds"] = preds_real
        all_preds.append(result)

        # ต่อค่าทำนายเข้า current_data เพื่อใช้ทำนายปีถัดไป
        new_rows = df_next[["province", "year"]].copy()
        new_rows["CO2_tonnes"] = preds_real
        current_data = pd.concat([current_data, new_rows], ignore_index=True)

    return pd.concat(all_preds, ignore_index=True)

# ─────────────────────────────────────────
# PIPELINE RUNNER (ปรับตาม logic ใหม่)
# ─────────────────────────────────────────
def run_pipeline(current_year_path: str, n_years: int = 1, start_year: int = None, end_year: int = None):
    print("── 1. Load current year data ──────────────────")
    df_current = load_and_clean(current_year_path)
    if "year" not in df_current.columns:
        import re
        match = re.search(r'(\d{4})', current_year_path)
        if match:
            current_year = int(match.group(1))
        else:
            current_year = 2024
        df_current["year"] = current_year
    current_year_val = df_current["year"].iloc[0]
    print(f"Current year: {current_year_val}")

    print("── 2. Load old data from old_data folder ──────")
    df_old = load_old_data(start_year, end_year)
    if df_old.empty:
        print("No old data found. Using only current year data.")
        combined = df_current.copy()
    else:
        print(f"Old data loaded: {len(df_old)} rows")
        combined = pd.concat([df_old, df_current], ignore_index=True)
        print(f"Combined data: {len(combined)} rows")

    print("── 3. Feature engineering ───────────")
    df_feat = add_features(combined)
    print("Years after add_features:", sorted(df_feat["year"].unique()))
    required_cols = ["province", "year", "CO2_tonnes", "lag1", "lag2", "growth", "rolling_mean"]
    df_feat = df_feat[required_cols]

    print("── 4. Encode & scale ────────────────")
    df_enc, encoder, scaler = encode_and_scale(df_feat)
    print("Years after encode_and_scale:", sorted(df_enc["year"].unique()))

    print("── 5. Anomaly detection ─────────────")
    df_enc = detect_anomalies(df_enc)

    print("── 6. Train & evaluate ──────────────")
    result, model ,mape,r2 = train_and_evaluate(df_enc, scaler, encoder)
    print("\nEvaluation Results (3 latest years):")
    print(result.to_string(index=False))

    print("\n── 7. Predict next year ──────────────")
    next_pred = predict_next_year(model, encoder, scaler, combined, n_years=n_years)
    print("\nPrediction for next year:")
    print(next_pred.to_string(index=False))

    print("\n── 8. Save current year to old_data ──────")
    save_current_to_old_data(df_current)
    historical_df = combined[["province", "year", "CO2_tonnes"]].copy()
    historical_df = historical_df.rename(columns={"CO2_tonnes": "actual"})

    return result, next_pred ,mape , r2,historical_df

def run_pipeline_without_current(n_years: int = 1, start_year: int = None, end_year: int = None):
    """
    Pipeline สำหรับกรณีไม่มีไฟล์ปัจจุบัน ใช้ข้อมูลจาก old_data.csv อย่างเดียว
    """
    print("── 1. Load old data only ──────────────────")
    df_old = load_old_data(start_year, end_year)
    if df_old.empty:
        raise ValueError("ไม่พบข้อมูลเก่าใน old_data.csv")

    print(f"Old data loaded: {len(df_old)} rows, years: {sorted(df_old['year'].unique())}")
    combined = df_old.copy()

    print("── 2. Feature engineering ───────────")
    df_feat = add_features(combined)
    required_cols = ["province", "year", "CO2_tonnes", "lag1", "lag2", "growth", "rolling_mean"]
    df_feat = df_feat[required_cols]

    print("── 3. Encode & scale ────────────────")
    df_enc, encoder, scaler = encode_and_scale(df_feat)

    print("── 4. Anomaly detection ─────────────")
    df_enc = detect_anomalies(df_enc)

    print("── 5. Train & evaluate ──────────────")
    result, model, mape, r2 = train_and_evaluate(df_enc, scaler, encoder)

    print("── 6. Predict next year(s) ──────────")
    next_pred = predict_next_year(model, encoder, scaler, combined, n_years=n_years)
    historical_df = combined[["province", "year", "CO2_tonnes"]].copy()
    historical_df = historical_df.rename(columns={"CO2_tonnes": "actual"})

    return result, next_pred, mape, r2 ,historical_df
# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = "2021.csv"   # ไฟล์ปีปัจจุบันที่ user input
    result, next_pred, model, encoder, scaler = run_pipeline(DATA_PATH)