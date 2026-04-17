import os
import tempfile
from fastapi import FastAPI,UploadFile,File,HTTPException,Query
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Dict,Any
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CO2 Prediction API", description="API สำหรับพยากรณ์ CO2 ปีถัดไป")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือระบุ origin จริง เช่น "http://localhost:5500"
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
async def predict_next_year_endpoint(file: UploadFile = File(...),n_years: int = Query(default=1, ge=1, le=10, description="จำนวนปีที่ต้องการทำนาย"),
    start_year: int = Query(None, description="ปีเริ่มต้นของข้อมูลเก่าที่ใช้เทรน (ค.ศ.)"),
    end_year: int = Query(None, description="ปีสิ้นสุดของข้อมูลเก่าที่ใช้เทรน (ค.ศ.)")):
    """
    รับไฟล์ CSV ของปีปัจจุบัน (ต้องมี columns: province, CO2_tonnes และ optionally year)
    คืนผลการทำนายปีถัดไป และผลการประเมิน (3 ปีล่าสุด)
    """
    # ตรวจสอบนามสกุลไฟล์
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV")

    try:
        # บันทึกไฟล์ที่อัปโหลดลง temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # เรียก pipeline
        # เนื่องจาก run_pipeline ในโค้ดเดิมใช้ print และคืนค่า result, next_pred, model, encoder, scaler
        # เราจำเป็นต้องมี run_pipeline อยู่ใน pipeline_main หรือสร้างฟังก์ชัน wrapper ใหม่
        # แต่วิธีที่ง่ายคือ copy run_pipeline มาไว้ที่นี่หรือ import จาก pipeline_main
        # สมมติว่ามี run_pipeline อยู่ใน pipeline_main แล้ว:
        from pipeline_main import run_pipeline
        # run_pipeline รับ path และคืน (result, next_pred, model, encoder, scaler)
        # result = evaluation DataFrame, next_pred = prediction DataFrame
        result_df, next_pred_df, mape, r2 = run_pipeline(
        tmp_path, 
        n_years=n_years,
        start_year=start_year,
        end_year=end_year
    )

        # ลบไฟล์ temp
        os.unlink(tmp_path)

        # ปรับ round ให้ทำทุกแถว (ทุกปี)
        next_pred_df["preds"] = (
            pd.to_numeric(next_pred_df["preds"], errors="coerce")
            .fillna(0).round(0).astype(int)
        )

        response = {
            "evaluation_summary": {"mape": float(mape), "r2": float(r2)},
            "prediction": next_pred_df.to_dict(orient="records"),
            "prediction_years": sorted(next_pred_df["year"].unique().tolist()),  # เปลี่ยนเป็น list
        }
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
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