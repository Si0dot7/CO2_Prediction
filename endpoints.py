import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse, Response
import pandas as pd
from typing import Optional
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from pipeline_main import (
    load_and_clean,
    load_old_data,
    save_current_to_old_data,
    add_features,
    encode_and_scale,
    detect_anomalies,
    train_and_evaluate,
    run_pipeline,
    run_pipeline_without_current,
)

from gnn_pipeline import run_gnn_pipeline, load_national_features
from process_elec import load_elec_profile
from thai_coor import THAILAND_PROVINCE_COORDS
from odiac_loader import load_odiac_for_pipeline, DEFAULT_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-load config — แก้ค่าตรงนี้ได้เลย
# ─────────────────────────────────────────────────────────────────────────────
ODIAC_FOLDER     = r"C:\Users\felm2\Desktop\sub\CO2_star_japan"
ODIAC_YEAR_START = 2000
ODIAC_YEAR_END   = 2022
EXTERNAL_FEAT_PATH = "./static/external.txt"   # ← path ของ external features
ELEC_CSV_PATH      = "./old_data/elec.csv"  # ← path ของ electricity data

# โหลด national features ครั้งเดียวตอน module load
try:
    NATIONAL_DF = load_national_features(EXTERNAL_FEAT_PATH)
    logger.info(f"[Config] โหลด external features สำเร็จ: "
                f"{len(NATIONAL_DF)} ปี ({NATIONAL_DF['year'].min()}–{NATIONAL_DF['year'].max()})")
except FileNotFoundError:
    NATIONAL_DF = None
    logger.warning(f"[Config] ไม่พบ {EXTERNAL_FEAT_PATH} — รันโดยไม่มี external features")

# โหลด electricity profile ครั้งเดียวตอน module load
try:
    _elec_raw = pd.read_csv(ELEC_CSV_PATH)
    ELEC_DF = _elec_raw
    logger.info(f"[Config] โหลด elec.csv สำเร็จ: {len(ELEC_DF)} แถว, "
                f"{ELEC_DF['ชื่อจังหวัด'].nunique()} จังหวัด")
except FileNotFoundError:
    ELEC_DF = None
    logger.warning(f"[Config] ไม่พบ {ELEC_CSV_PATH} — รันโดยไม่มี electricity static features")

app = FastAPI(
    title="CO2 Prediction API",
    description="API สำหรับพยากรณ์ CO2 ปีถัดไป (XGBoost + ST-GNN)",
)


@app.on_event("startup")
async def auto_load_odiac():
    """
    โหลดข้อมูล ODIAC อัตโนมัติตอน server เริ่ม
    ถ้า old_data.csv มีอยู่แล้ว จะข้ามการโหลดซ้ำ
    """
    old_data_path = os.path.join("old_data", "old_data.csv")
    if os.path.exists(old_data_path):
        logger.info(f"[Startup] พบ old_data.csv อยู่แล้ว — ข้ามการโหลด ODIAC")
        return

    logger.info(f"[Startup] ไม่พบ old_data.csv — เริ่มโหลด ODIAC จาก {ODIAC_FOLDER}")
    try:
        df = load_odiac_for_pipeline(
            folder=ODIAC_FOLDER,
            year_start=ODIAC_YEAR_START,
            year_end=ODIAC_YEAR_END,
        )
        from pipeline_main import save_current_to_old_data
        save_current_to_old_data(df)
        logger.info(f"[Startup] โหลดสำเร็จ: {len(df)} แถว, "
                    f"{df['province'].nunique()} จังหวัด, "
                    f"ปี {df['year'].min()}–{df['year'].max()}")
    except Exception as e:
        logger.error(f"[Startup] โหลด ODIAC ไม่สำเร็จ: {e}")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
async def _load_df_from_upload(file: UploadFile) -> tuple:
    """บันทึกไฟล์ชั่วคราว → ตรวจสอบปี → คืน (df, tmp_path)"""
    import re
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be CSV")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    df = pd.read_csv(tmp_path)

    if "year" in df.columns:
        years_in_file = df["year"].unique()
    else:
        m = re.search(r"(\d{4})", file.filename)
        if m:
            years_in_file = [int(m.group(1))]
        else:
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=400,
                detail="ไม่พบปีในไฟล์ กรุณาระบุคอลัมน์ 'year' หรือใส่ปีในชื่อไฟล์",
            )

    if max(years_in_file) < 2023:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400,
                            detail="ไฟล์ต้องมีข้อมูลปี 2023 ขึ้นไปเท่านั้น")

    return df, tmp_path


def _format_preds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["preds"] = (pd.to_numeric(df["preds"], errors="coerce")
                   .fillna(0).round(0).astype(int))
    return df


def _check_province_coords(provinces: list) -> list:
    return [p for p in provinces if p not in THAILAND_PROVINCE_COORDS]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1: XGBoost
# ─────────────────────────────────────────────────────────────────────────────
# @app.post("/predict", tags=["XGBoost"])
# async def predict_xgboost(
#     file: Optional[UploadFile] = File(None),
#     n_years: int  = Query(default=1, ge=1, le=10),
#     start_year: int = Query(None),
#     end_year: int   = Query(None),
# ):
#     """ทำนาย CO2 ด้วย XGBoost"""
#     tmp_path = None
#     try:
#         if file is not None:
#             _, tmp_path = await _load_df_from_upload(file)
#             result_df, next_pred_df, mape, r2, historical_df = run_pipeline(
#                 tmp_path, n_years=n_years, start_year=start_year, end_year=end_year)
#         else:
#             result_df, next_pred_df, mape, r2, historical_df = run_pipeline_without_current(
#                 n_years=n_years, start_year=start_year, end_year=end_year)

#         next_pred_df = _format_preds(next_pred_df)
#         return JSONResponse(content={
#             "model": "xgboost",
#             "evaluation_summary": {"mape": float(mape), "r2": float(r2)},
#             "prediction": next_pred_df.to_dict(orient="records"),
#             "historical": historical_df.to_dict(orient="records"),
#             "prediction_years": sorted(next_pred_df["year"].unique().tolist()),
#         })

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"XGBoost error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2: ST-GNN
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict/gnn", tags=["GNN"])
async def predict_gnn(
    file: Optional[UploadFile] = File(None),
    n_years: int     = Query(default=1, ge=1, le=10),
    start_year: int  = Query(None),
    end_year: int    = Query(None),
    k_neighbors: int = Query(default=5, ge=1, le=20),
    seq_len: int     = Query(default=8, ge=2, le=15),
    epochs: int      = Query(default=300, ge=50, le=1000),
    hidden_dim: int  = Query(default=64, ge=16, le=256),
    use_ensemble: bool = Query(default=True, description="ใช้ ensemble 3 models (แม่นกว่า แต่ช้ากว่า)"),
    n_models: int    = Query(default=3, ge=1, le=5, description="จำนวน models ใน ensemble"),
):
    """
    ทำนาย CO2 ด้วย ST-GNN (Graph WaveNet + Ensemble)
    - use_ensemble=True: ช้ากว่า แต่แม่นกว่า (แนะนำ)
    - use_ensemble=False: เร็วกว่า สำหรับ prototype/debug
    """
    tmp_path = None
    try:
        # โหลดข้อมูล
        if file is not None:
            _, tmp_path = await _load_df_from_upload(file)
            df_current = load_and_clean(tmp_path)
            if "year" not in df_current.columns:
                import re
                m = re.search(r"(\d{4})", file.filename)
                df_current["year"] = int(m.group(1)) if m else 2024
            df_old = load_old_data(start_year, end_year)
            combined = (pd.concat([df_old, df_current], ignore_index=True)
                        if not df_old.empty else df_current.copy())
            save_current_to_old_data(df_current)
        else:
            combined = load_old_data(start_year, end_year)
            if combined.empty:
                raise HTTPException(status_code=400, detail="ไม่พบข้อมูลใน old_data.csv")

        # ตรวจสอบข้อมูลขั้นต่ำ
        n_years_data = combined["year"].nunique()
        min_required = seq_len + 3
        if n_years_data < min_required:
            raise HTTPException(
                status_code=400,
                detail=f"ข้อมูลน้อยเกินไป: มี {n_years_data} ปี ต้องการอย่างน้อย {min_required} ปี")

        # ตรวจสอบ coords
        provinces_in_data = combined["province"].unique().tolist()
        missing = _check_province_coords(provinces_in_data)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"ไม่พบ lat/lon: {missing}. ชื่อจังหวัดต้องเป็นภาษาอังกฤษ")

        logger.info(f"GNN: {n_years_data} years, {len(provinces_in_data)} provinces, "
                    f"ensemble={use_ensemble}, n_models={n_models}")

        result_df, next_pred_df, mape, r2, historical_df = run_gnn_pipeline(
            df=combined,
            province_coords=THAILAND_PROVINCE_COORDS,
            n_years=n_years,
            k_neighbors=k_neighbors,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=1e-3,
            dropout=0.3,
            test_years=3,
            use_ensemble=use_ensemble,
            n_models=n_models,
            national_df=NATIONAL_DF,
            elec_df=ELEC_DF,
        )

        next_pred_df = _format_preds(next_pred_df)
        return JSONResponse(content={
            "model": "st_gnn_wavenet",
            "evaluation_summary": {"mape": float(mape), "r2": float(r2)},
            "prediction": next_pred_df.to_dict(orient="records"),
            "historical": historical_df.to_dict(orient="records"),
            "prediction_years": sorted(next_pred_df["year"].unique().tolist()),
            "graph_info": {
                "num_nodes": len(provinces_in_data),
                "k_neighbors": k_neighbors,
                "seq_len": seq_len,
                "ensemble": use_ensemble,
                "n_models": n_models if use_ensemble else 1,
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GNN error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GNN error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3: Benchmark XGBoost vs GNN
# ─────────────────────────────────────────────────────────────────────────────
# @app.post("/predict/benchmark", tags=["Benchmark"])
# async def benchmark(
#     file: Optional[UploadFile] = File(None),
#     start_year: int  = Query(None),
#     end_year: int    = Query(None),
#     seq_len: int     = Query(default=8, ge=2, le=15),
#     epochs: int      = Query(default=300, ge=50, le=500),
#     use_ensemble: bool = Query(default=True),
# ):
#     """รัน XGBoost และ GNN พร้อมกัน แล้วเปรียบเทียบ MAPE และ R²"""
#     tmp_path = None
#     try:
#         if file is not None:
#             _, tmp_path = await _load_df_from_upload(file)
#             df_current = load_and_clean(tmp_path)
#             if "year" not in df_current.columns:
#                 import re
#                 m = re.search(r"(\d{4})", file.filename)
#                 df_current["year"] = int(m.group(1)) if m else 2024
#             df_old = load_old_data(start_year, end_year)
#             combined = (pd.concat([df_old, df_current], ignore_index=True)
#                         if not df_old.empty else df_current.copy())
#         else:
#             combined = load_old_data(start_year, end_year)
#             if combined.empty:
#                 raise HTTPException(status_code=400, detail="ไม่พบข้อมูลใน old_data.csv")

#         results = {}

#         # XGBoost
#         logger.info("Benchmark: XGBoost...")
#         try:
#             df_feat = add_features(combined.copy())
#             required = ["province", "year", "CO2_tonnes", "lag1", "lag2",
#                         "lag3", "rolling_mean_3", "rolling_std_3", "growth"]
#             df_enc, encoder, scaler = encode_and_scale(df_feat[required])
#             df_enc = detect_anomalies(df_enc, contamination=0.01)
#             _, _, xgb_mape, xgb_r2 = train_and_evaluate(df_enc, scaler, encoder)
#             results["xgboost"] = {"mape": float(xgb_mape), "r2": float(xgb_r2), "status": "success"}
#         except Exception as e:
#             results["xgboost"] = {"status": "failed", "error": str(e)}

#         # GNN
#         logger.info("Benchmark: ST-GNN...")
#         try:
#             provinces_in_data = combined["province"].unique().tolist()
#             missing = _check_province_coords(provinces_in_data)
#             if missing:
#                 results["gnn"] = {"status": "failed", "error": f"Missing coords: {missing}"}
#             else:
#                 _, _, gnn_mape, gnn_r2, _ = run_gnn_pipeline(
#                     df=combined,
#                     province_coords=THAILAND_PROVINCE_COORDS,
#                     n_years=1, k_neighbors=5, seq_len=seq_len,
#                     hidden_dim=64, epochs=epochs, test_years=3,
#                     dropout=0.2, use_ensemble=use_ensemble, n_models=3,
#                 )
#                 results["gnn"] = {"mape": float(gnn_mape), "r2": float(gnn_r2), "status": "success"}
#         except Exception as e:
#             results["gnn"] = {"status": "failed", "error": str(e)}

#         winner = None
#         if (results.get("xgboost", {}).get("status") == "success" and
#                 results.get("gnn", {}).get("status") == "success"):
#             winner = ("xgboost" if results["xgboost"]["mape"] < results["gnn"]["mape"]
#                       else "gnn")

#         return JSONResponse(content={
#             "benchmark": results,
#             "winner_by_mape": winner,
#             "note": "MAPE ต่ำกว่า = ดีกว่า | R² สูงกว่า = ดีกว่า",
#         })

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Benchmark error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 4: โหลดข้อมูล ODIAC จากโฟลเดอร์ (10km resolution)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/data/load-odiac", tags=["Data"])
async def load_odiac_data(
    folder:     str           = Query(default=DEFAULT_FOLDER,
                                      description="path โฟลเดอร์ที่มีไฟล์ .xyz"),
    year_start: Optional[int] = Query(None, description="ปีเริ่มต้น เช่น 2000"),
    year_end:   Optional[int] = Query(None, description="ปีสิ้นสุด เช่น 2022"),
    save_csv:   bool          = Query(default=True,
                                      description="บันทึก old_data.csv ให้ pipeline ใช้ต่อ"),
):
    """
    อ่านข้อมูล ODIAC .xyz จากโฟลเดอร์บนเครื่อง
    แปลง resolution 1km → 10km และ aggregate เป็นรายจังหวัด-รายปี
    สามารถบันทึกเป็น old_data.csv เพื่อใช้กับ /predict และ /predict/gnn ได้ทันที
    """
    try:
        df = load_odiac_for_pipeline(
            folder=folder,
            year_start=year_start,
            year_end=year_end,
        )

        if save_csv:
            from pipeline_main import save_current_to_old_data
            save_current_to_old_data(df)
            logger.info("บันทึก old_data.csv เรียบร้อย")

        missing_prov = _check_province_coords(df["province"].unique().tolist())

        return JSONResponse(content={
            "status": "ok",
            "rows": len(df),
            "provinces": df["province"].nunique(),
            "years": sorted(df["year"].unique().tolist()),
            "year_range": {"min": int(df["year"].min()), "max": int(df["year"].max())},
            "co2_summary": {
                "mean":  round(float(df["CO2_tonnes"].mean()), 2),
                "min":   round(float(df["CO2_tonnes"].min()),  2),
                "max":   round(float(df["CO2_tonnes"].max()),  2),
                "total": round(float(df["CO2_tonnes"].sum()),  2),
            },
            "saved_to_old_data": save_csv,
            "missing_coords": missing_prov,
            "sample": df.head(10).to_dict(orient="records"),
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"ODIAC load error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    import torch
    return {"status": "ok", "cuda_available": torch.cuda.is_available()}


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.head("/")
async def head_root():
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn, asyncio, sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)