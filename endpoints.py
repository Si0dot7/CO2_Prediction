import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Dict, Any, Optional
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from pipeline_main import (
    load_and_clean,
    load_old_data,
    save_current_to_old_data,
    add_features,
    encode_and_scale,
    detect_anomalies,
    train_and_evaluate,
    predict_next_year,
    run_pipeline,
    run_pipeline_without_current   # <-- เพิ่ม import
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CO2 Prediction API", description="API สำหรับพยากรณ์ CO2 ปีถัดไป")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=500)

class PredictionResponse:
    def __init__(self, next_year_pred: pd.DataFrame, evaluation_result: pd.DataFrame):
        self.next_year_pred = next_year_pred.to_dict(orient="records")
        self.evaluation_result = evaluation_result.to_dict(orient="records") if not evaluation_result.empty else []

@app.post("/predict")
async def predict_next_year_endpoint(
    file: Optional[UploadFile] = File(None),
    n_years: int = Query(default=1, ge=1, le=10, description="จำนวนปีที่ต้องการทำนาย"),
    start_year: int = Query(None, description="ปีเริ่มต้นของข้อมูลเก่าที่ใช้เทรน (ค.ศ.)"),
    end_year: int = Query(None, description="ปีสิ้นสุดของข้อมูลเก่าที่ใช้เทรน (ค.ศ.)")
):
    """
    รับไฟล์ CSV ของปีปัจจุบัน (ต้องมี columns: province, CO2_tonnes และ optionally year)
    หรือถ้าไม่ส่งไฟล์มา จะใช้ข้อมูลเก่าจาก old_data.csv ทั้งหมด
    คืนผลการทำนายปีถัดไป และผลการประเมิน
    """
    tmp_path = None
    try:
        # กรณีมีไฟล์อัปโหลด
        if file is not None:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="File must be CSV")

            # บันทึกไฟล์ชั่วคราว
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # ตรวจสอบว่าข้อมูลในไฟล์เป็นปี >= 2023
            df_check = pd.read_csv(tmp_path)
            if "year" in df_check.columns:
                years_in_file = df_check["year"].unique()
            else:
                # ถ้าไม่มีคอลัมน์ year ให้ลองหาจากชื่อไฟล์
                import re
                match = re.search(r'(\d{4})', file.filename)
                if match:
                    years_in_file = [int(match.group(1))]
                else:
                    raise HTTPException(status_code=400, detail="ไม่พบปีในไฟล์ กรุณาระบุคอลัมน์ 'year' หรือใส่ปีในชื่อไฟล์")
            if max(years_in_file) < 2023:
                raise HTTPException(status_code=400, detail="ไฟล์ต้องมีข้อมูลปี 2023 ขึ้นไปเท่านั้น")

            # เรียก pipeline ปกติ (ใช้ไฟล์ปัจจุบัน + old_data)
            result_df, next_pred_df, mape, r2 = run_pipeline(
                tmp_path,
                n_years=n_years,
                start_year=start_year,
                end_year=end_year
            )
        else:
            # ไม่มีไฟล์ -> ใช้ old_data.csv อย่างเดียว
            result_df, next_pred_df, mape, r2 , historical_df = run_pipeline_without_current(
                n_years=n_years,
                start_year=start_year,
                end_year=end_year
            )

        # จัดการค่าทำนายให้เป็นจำนวนเต็ม
        next_pred_df["preds"] = (
            pd.to_numeric(next_pred_df["preds"], errors="coerce")
            .fillna(0).round(0).astype(int)
        )

        response = {
            "evaluation_summary": {"mape": float(mape), "r2": float(r2)},
            "prediction": next_pred_df.to_dict(orient="records"),
            "historical": historical_df.to_dict(orient="records"),
            "prediction_years": sorted(next_pred_df["year"].unique().tolist()),
        }
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=8000)