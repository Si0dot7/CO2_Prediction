# ระบบคาดการณ์ CO₂ รายจังหวัด — Project README

> อ่านไฟล์นี้ก่อนเริ่มทุกครั้ง เพื่อเข้าใจโครงสร้างโปรเจ็คโดยไม่ต้องดูโค้ดทีละไฟล์

---

## 1. ภาพรวม

ระบบพยากรณ์ปริมาณการปล่อย CO₂ รายจังหวัด 77 จังหวัดในประเทศไทย
ใช้ข้อมูลจาก ODIAC satellite (1km resolution) เป็น target และมีปัจจัยเสริมจาก GPP + ไฟฟ้า + external national features

- **Model หลัก:** Graph WaveNet ST-GNN v8 (Spatio-Temporal Graph Neural Network)
- **Model สำรอง:** XGBoost (commented out ใน endpoints แต่โค้ดยังอยู่)
- **Frontend:** Single-page app (Vanilla JS + Chart.js + D3 + Bootstrap 5)
- **Backend:** FastAPI (Python) + Uvicorn
- **ภาษา UI:** ภาษาไทย

---

## 2. โครงสร้างไฟล์

```
project/
├── endpoints.py          # FastAPI app — entry point หลัก
├── gnn_pipeline.py       # GNN model + training + inference (v8)
├── pipeline_main.py      # XGBoost pipeline (ใช้ใน endpoints แต่ไม่ได้ส่งมา)
├── odiac_loader.py       # โหลด ODIAC .xyz → DataFrame รายจังหวัด/ปี
├── gpp_process.py        # โหลด GPP Excel (NESDC) → long/wide format
├── process_elec.py       # โหลด Excel ไฟฟ้า → elec.csv + load_elec_profile()
├── thai_coor.py          # dict ชื่อจังหวัด EN → (lat, lon) centroid 77 จังหวัด
│
├── static/
│   ├── index.html        # Frontend หลัก
│   ├── app.js            # Frontend logic ทั้งหมด
│   ├── style.css         # CSS (Bootstrap custom theme สีเขียวรัฐบาล + gold)
│   ├── external.txt      # National CO₂ data แยก sector (ปัจจัยภายนอก)
│   ├── edgar.txt         # EDGAR CO₂ reference data (toggle ใน chart)
│   └── gca.txt           # GCA CO₂ reference data (toggle ใน chart)
│
└── old_data/
    ├── old_data.csv      # ODIAC data รายปี-จังหวัด (สร้างโดย odiac_loader)
    ├── elec.csv          # ข้อมูลไฟฟ้า (สร้างโดย process_elec.py)
    ├── gpp_long.csv      # GPP long format (สร้างโดย gpp_process.py)
    └── gpp_wide.csv      # GPP wide format (pivot)
```

---

## 3. Data Sources

| ไฟล์/Data | แหล่งที่มา | รูปแบบ | ช่วงปี |
|---|---|---|---|
| `old_data.csv` | ODIAC satellite `.xyz` | province, year, CO2_tonnes | 2000–2022 |
| `elec.csv` | MEA/PEA Excel รายปี | ชื่อจังหวัด(TH), year, 4 cols | 2018–2022 |
| `gpp_long.csv` | NESDC GPP Excel | province(EN), year, sector, value | 1995–2024 |
| `external.txt` | รวบรวมระดับประเทศ | CSV, year + 7 cols | 2000–2022 |
| `edgar.txt` / `gca.txt` | EDGAR / GCA | space-separated, year val | ตามแหล่ง |

### external.txt columns (ปัจจัยภายนอก — ใช้ทั้ง frontend chart และ GNN features)
```
year, energy_transport, industrial_processes, agriculture, waste,
forestry_land_use,   ← ค่าติดลบ = ป่าไม้ดูดซับ CO₂
total_excl_forestry, total_incl_forestry
```
> GNN v8 ตัด `total_excl_forestry` ออกเพื่อป้องกัน data leakage
> และสร้าง derived features: `transport_share`, `forestry_growth`, `industrial_growth`, `transport_growth`

---

## 4. GNN Model (v8) — สิ่งสำคัญ

**Architecture:** Graph WaveNet + Adaptive Adjacency + Ensemble

| Component | รายละเอียด |
|---|---|
| Input features | 7 temporal (scaled CO2, lag1–3, rolling3, growth, year_idx) + 4 elec static = **11 channels** |
| Graph | k-NN adjacency บน lat/lon (Haversine) + Adaptive Adjacency (learnable) |
| Global injection | National features → MLP → per-node weight (node-aware, init จาก industrial_elec) |
| Output | CO2_tonnes ต่อจังหวัด ต่อปี |
| Ensemble | 3–5 models, ค่าเฉลี่ย prediction |

**Parameters ที่ปรับได้ใน API:**
- `k_neighbors` (default 5) — ขนาด graph
- `seq_len` (default 8) — lookback window
- `epochs` (default 300)
- `hidden_dim` (default 64)
- `use_ensemble` (default True) — ช้ากว่าแต่แม่นกว่า
- `n_models` (default 3)

---

## 5. API Endpoints

| Method | Path | คำอธิบาย |
|---|---|---|
| `POST` | `/predict/gnn` | ทำนาย CO₂ ด้วย ST-GNN หลัก |
| `POST` | `/data/load-odiac` | โหลด ODIAC .xyz folder → old_data.csv |
| `GET` | `/health` | ตรวจสอบ server + CUDA |
| `GET` | `/` | redirect → `/static/index.html` |

**`/predict/gnn` request:**
- `file` (optional): CSV ปี 2023+ format เดียวกับ old_data.csv
- Query params: `n_years`, `start_year`, `end_year`, `k_neighbors`, `seq_len`, `epochs`, `hidden_dim`, `use_ensemble`, `n_models`

**`/predict/gnn` response:**
```json
{
  "model": "st_gnn_wavenet",
  "evaluation_summary": {"mape": 0.xx, "r2": 0.xx},
  "prediction": [{"province": "...", "year": 2023, "preds": 123456}, ...],
  "historical": [{"province": "...", "year": 2000, "actual": 123456}, ...],
  "graph_info": {...}
}
```

---

## 6. Frontend (app.js + index.html)

### ฟีเจอร์หลัก
1. **Upload & Predict** — กรอก n_years, เลือก model, upload CSV (optional)
2. **ตารางผล** — รายจังหวัด/ปี พร้อม summary cards
3. **กราฟ province** — เส้นแยกจังหวัด, checkbox เลือก, search, Top 5
4. **กราฟรวมประเทศ** — ประวัติ + คาดการณ์ + toggle EDGAR/GCA overlay
5. **กราฟปัจจัยภายนอก** — Stacked Bar (ภาคปล่อย + ป่าไม้ดูดซับ) + เส้น net total
6. **Map Presentation Mode** — fullscreen D3 choropleth แผนที่ไทย + KPI + ranking

### State variables สำคัญ (app.js)
```javascript
currentPredictionData   // [{province, year, preds}]
currentHistoricalData   // [{province, year, actual}]
allYearsList            // sorted year array (hist + pred)
firstPredYearValue      // ปีแรกที่เป็น prediction (แบ่งเส้นทึบ/ประ)
globalMaxPred           // max preds value (ใช้ color scale แผนที่)
externalRawData         // parsed external.txt rows
edgarData / gcaData     // {year: CO2_tonnes}
```

### ชื่อ mapping ที่ใช้ใน frontend
- `EN2TH` — ชื่อ EN → ชื่อไทย (77 จังหวัด)
- `API_TO_GEOJSON` — ชื่อ API (EN) → ชื่อใน GeoJSON (บางตัวต่างกัน เช่น Bangkok → Bangkok Metropolis)
- GeoJSON source: `https://raw.githubusercontent.com/apisit/thailand.json/master/thailand.json`

---

## 7. Province Name Convention

**ชื่อมาตรฐาน** ที่ใช้ใน backend คือ English ตาม `thai_coor.py` (77 entries)
ไฟล์ต่างๆ ต้องใช้ชื่อตรงกับ key ใน dict นั้น

ตัวอย่างชื่อที่เคยมี mismatch:
| ชื่อในข้อมูลดิบ | ชื่อมาตรฐาน |
|---|---|
| Bangkok Metropolis (GPP) | Bangkok |
| Buri Ram (GPP) | Buriram |
| Chai Nat (GPP) | Chainat |
| Lop Buri (GPP) | Lopburi |
| Si Sa Ket (GPP) | Sisaket |
| Chon Buri (GPP) | Chonburi |

Mapping เต็มอยู่ใน `gpp_process.py > GPP_TO_ODIAC`

---

## 8. การรัน Server

```bash
# ติดตั้ง dependencies
pip install fastapi uvicorn torch pandas scikit-learn scipy openpyxl

# รัน (Windows)
python endpoints.py

# หรือ
uvicorn endpoints:app --host 0.0.0.0 --port 8000 --reload
```

**config ที่ต้องแก้ใน `endpoints.py`:**
```python
ODIAC_FOLDER     = r"C:\Users\felm2\Desktop\sub\CO2_star_japan"  # ← path ไฟล์ .xyz
ODIAC_YEAR_START = 2000
ODIAC_YEAR_END   = 2022
EXTERNAL_FEAT_PATH = "./static/external.txt"
ELEC_CSV_PATH      = "./old_data/elec.csv"
```

**Startup behavior:** ถ้าไม่มี `old_data/old_data.csv` → โหลด ODIAC อัตโนมัติจาก `ODIAC_FOLDER`

---

## 9. การเตรียมข้อมูล (ครั้งแรก)

```python
# 1. สร้าง old_data.csv จาก ODIAC .xyz
from odiac_loader import load_odiac_for_pipeline
df = load_odiac_for_pipeline(folder="path/to/xyz", year_start=2000, year_end=2022)
# → บันทึกอัตโนมัติผ่าน API /data/load-odiac หรือ startup

# 2. สร้าง elec.csv จาก Excel ไฟฟ้า
# แก้ INPUT_FOLDER ใน process_elec.py แล้วรัน
python process_elec.py

# 3. สร้าง gpp_long.csv / gpp_wide.csv
# แก้ EXCEL_PATH ใน gpp_process.py แล้วรัน
python gpp_process.py
```

**elec.csv format:** `ชื่อจังหวัด(TH), year, industrial_electricity, residential_electricity, public_electricity, agriculture_electricity`

**old_data.csv format:** `province(EN), year, CO2_tonnes`

---

## 10. Known Issues / Notes

- **XGBoost endpoint** (`/predict`) ถูก comment out ใน endpoints.py — โค้ดอยู่ใน `pipeline_main.py`
- **Benchmark endpoint** (`/predict/benchmark`) ก็ถูก comment out เช่นกัน
- `process_elec.py` มี `public_electricity` ที่บางปีเป็น NaN → fill ด้วย 0 อัตโนมัติ
- GNN ต้องการข้อมูลอย่างน้อย `seq_len + 3` ปี (default = 11 ปี)
- `forestry_land_use` ใน external.txt เป็น **ค่าติดลบ** (ดูดซับ CO₂) — แสดงใต้แกนศูนย์ในกราฟ
- CUDA ใช้อัตโนมัติถ้ามี GPU — ตรวจสอบได้ที่ `/health`
