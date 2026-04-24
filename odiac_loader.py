"""
odiac_loader.py
────────────────────────────────────────────────────────────────────────────
ดึงข้อมูล ODIAC .xyz files → แปลง resolution 1km → 10km
→ aggregate เป็นรายจังหวัด → คืน DataFrame ที่ใช้กับ pipeline ได้ทันที

OUTPUT DataFrame columns:
    province   : str   (ชื่อจังหวัด ภาษาอังกฤษ ตรงกับ thai_coor.py)
    year       : int
    month      : int
    CO2_tonnes : float (tonne C / ปี ต่อพื้นที่รวมของจังหวัด)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

# ── import พิกัดจังหวัด ─────────────────────────────────────────────────────
try:
    from thai_coor import THAILAND_PROVINCE_COORDS
except ImportError:
    raise ImportError(
        "ไม่พบ thai_coor.py — วางไฟล์ไว้ใน directory เดียวกับ odiac_loader.py"
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# ODIAC unit: gC / m²/ day  →  ต้องการ tonne C / cell / year
# พื้นที่ cell 1 km²  = 1e6 m²
# 1 gC = 1e-6 tonne C
# วันต่อเดือน: ใช้ค่าเฉลี่ยต่อปี (365.25 / 12)
DAYS_PER_MONTH = 365.25 / 12         # ~30.44
M2_PER_KM2    = 1e6
GC_TO_TONNE   = 1e-6

# ขอบเขตประเทศไทย (Bounding Box) — ตัดข้อมูลที่ไม่ใช่ไทยออกก่อน aggregate
THAI_LAT_MIN, THAI_LAT_MAX = 5.5,  20.5
THAI_LON_MIN, THAI_LON_MAX = 97.5, 105.7

# resolution เป้าหมาย (องศา)
TARGET_RES_DEG = 10 / 111.0   # ~0.09009°  (10 km ≈ 111 km/°)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ค้นหาและ parse ชื่อไฟล์
# ─────────────────────────────────────────────────────────────────────────────

def _parse_filename(fname: str) -> tuple[int, int] | None:
    """
    แยก year/month จากชื่อไฟล์
    ShpCut_BoundTH_odiac2023_1km_excl_intl_YYMM.xyz
    4 ตัวสุดท้ายก่อน .xyz  เช่น 0001 → year=2000, month=1
                                       2212 → year=2022, month=12
    """
    m = re.search(r"_(\d{4})\.xyz$", fname, re.IGNORECASE)
    if not m:
        return None
    code = m.group(1)
    yy   = int(code[:2])   # 2 ตัวแรก = YY
    mm   = int(code[2:])   # 2 ตัวหลัง = MM
    if not (1 <= mm <= 12):
        return None
    year = 2000 + yy
    return year, mm


def list_xyz_files(folder: str) -> list[dict]:
    """
    สแกนโฟลเดอร์ คืน list ของ {path, year, month}
    """
    pattern = os.path.join(folder, "*.xyz")
    files   = sorted(glob.glob(pattern))
    result  = []
    for fp in files:
        parsed = _parse_filename(os.path.basename(fp))
        if parsed:
            result.append({"path": fp, "year": parsed[0], "month": parsed[1]})
    logger.info(f"พบ {len(result)} ไฟล์ .xyz ใน {folder}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. โหลดและแปลง .xyz → DataFrame ระดับ pixel
# ─────────────────────────────────────────────────────────────────────────────

def _load_xyz(path: str) -> pd.DataFrame:
    """
    อ่าน .xyz file (whitespace-separated: lon lat value)
    กรองเฉพาะ bounding box ประเทศไทย
    แปลง gC/m²/day → tonne C / cell (ต่อเดือน)
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "c"],
        dtype=np.float32,
        comment="#",
    )

    # กรองพิกัดนอกไทย
    mask = (
        (df["lat"] >= THAI_LAT_MIN) & (df["lat"] <= THAI_LAT_MAX) &
        (df["lon"] >= THAI_LON_MIN) & (df["lon"] <= THAI_LON_MAX) &
        (df["c"]   >  0)
    )
    df = df[mask].copy()

    df["co2_tonne"] = df["c"] * 3.664

    return df[["lat", "lon", "co2_tonne"]]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Resample 1 km → 10 km
# ─────────────────────────────────────────────────────────────────────────────

def _resample_to_10km(df: pd.DataFrame) -> pd.DataFrame:
    """
    bin pixels เข้า grid 10 km × 10 km (sum ค่า CO2 ใน bin)
    คืน DataFrame: lat_center, lon_center, co2_tonne
    """
    if df.empty:
        return df

    lat_bins = np.arange(THAI_LAT_MIN, THAI_LAT_MAX + TARGET_RES_DEG, TARGET_RES_DEG)
    lon_bins = np.arange(THAI_LON_MIN, THAI_LON_MAX + TARGET_RES_DEG, TARGET_RES_DEG)

    stat, lat_edges, lon_edges, _ = binned_statistic_2d(
        df["lat"].values,
        df["lon"].values,
        df["co2_tonne"].values,
        statistic="sum",
        bins=[lat_bins, lon_bins],
    )

    # แปลง matrix → DataFrame (เฉพาะ cell ที่ไม่ใช่ NaN)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    rows = []
    lat_idx, lon_idx = np.where(~np.isnan(stat) & (stat > 0))
    rows = {
        "lat": lat_centers[lat_idx],
        "lon": lon_centers[lon_idx],
        "co2_tonne": stat[lat_idx, lon_idx],
    }
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assign จังหวัด (nearest-centroid)
# ─────────────────────────────────────────────────────────────────────────────

def _build_province_arrays() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """สร้าง array lat/lon ของ centroid จังหวัดจาก thai_coor.py"""
    names = list(THAILAND_PROVINCE_COORDS.keys())
    lats  = np.array([THAILAND_PROVINCE_COORDS[n][0] for n in names])
    lons  = np.array([THAILAND_PROVINCE_COORDS[n][1] for n in names])
    return lats, lons, names


_PROV_LATS, _PROV_LONS, _PROV_NAMES = _build_province_arrays()


def _assign_province_vectorized(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    กำหนดจังหวัดให้แต่ละ pixel/cell ด้วย nearest-centroid
    ใช้ Euclidean บน lat/lon (เพียงพอสำหรับขนาดประเทศไทย)
    """
    # broadcast: (n_cells, n_provinces)
    dlat = lats[:, None] - _PROV_LATS[None, :]
    dlon = lons[:, None] - _PROV_LONS[None, :]
    dist2 = dlat**2 + dlon**2
    idx   = np.argmin(dist2, axis=1)
    return np.array(_PROV_NAMES)[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Process ไฟล์เดียว → แถวรายจังหวัด
# ─────────────────────────────────────────────────────────────────────────────

def process_single_file(path: str, year: int, month: int) -> pd.DataFrame:
    """
    โหลด 1 ไฟล์ → resample → aggregate ตามจังหวัด
    คืน DataFrame: province, year, month, CO2_tonnes
    """
    raw     = _load_xyz(path)
    if raw.empty:
        logger.warning(f"ไม่มีข้อมูลใน {path}")
        return pd.DataFrame(columns=["province", "year", "month", "CO2_tonnes"])

    resampled = _resample_to_10km(raw)

    if resampled.empty:
        return pd.DataFrame(columns=["province", "year", "month", "CO2_tonnes"])

    resampled["province"] = _assign_province_vectorized(
        resampled["lat"].values, resampled["lon"].values
    )

    agg = (
        resampled
        .groupby("province", as_index=False)["co2_tonne"]
        .sum()
        .rename(columns={"co2_tonne": "CO2_tonnes"})
    )
    agg["year"]  = year
    agg["month"] = month
    return agg[["province", "year", "month", "CO2_tonnes"]]


# ─────────────────────────────────────────────────────────────────────────────
# 6. โหลดทั้งโฟลเดอร์ → DataFrame รวม
# ─────────────────────────────────────────────────────────────────────────────

def load_odiac_folder(
    folder: str,
    year_start: Optional[int] = None,
    year_end:   Optional[int] = None,
    aggregate_monthly: bool = True,
) -> pd.DataFrame:
    """
    โหลดข้อมูล ODIAC ทั้งโฟลเดอร์

    Parameters
    ----------
    folder            : path โฟลเดอร์ที่มีไฟล์ .xyz
    year_start        : กรองเฉพาะปี >= year_start (None = ทั้งหมด)
    year_end          : กรองเฉพาะปี <= year_end   (None = ทั้งหมด)
    aggregate_monthly : True  → sum รายเดือน → รายปี (CO2_tonnes ต่อปี ต่อจังหวัด)
                        False → คืนข้อมูลรายเดือน

    Returns
    -------
    pd.DataFrame
        aggregate_monthly=True  → columns: province, year, CO2_tonnes
        aggregate_monthly=False → columns: province, year, month, CO2_tonnes
    """
    file_list = list_xyz_files(folder)
    if not file_list:
        raise FileNotFoundError(f"ไม่พบไฟล์ .xyz ใน {folder}")

    # กรองตามช่วงปี
    if year_start is not None:
        file_list = [f for f in file_list if f["year"] >= year_start]
    if year_end is not None:
        file_list = [f for f in file_list if f["year"] <= year_end]

    if not file_list:
        raise ValueError(f"ไม่มีไฟล์ในช่วงปี {year_start}–{year_end}")

    logger.info(f"ประมวลผล {len(file_list)} ไฟล์ (ปี {file_list[0]['year']}–{file_list[-1]['year']})")

    parts = []
    for i, info in enumerate(file_list, 1):
        logger.info(f"  [{i}/{len(file_list)}] {os.path.basename(info['path'])} "
                    f"({info['year']}/{info['month']:02d})")
        part = process_single_file(info["path"], info["year"], info["month"])
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)

    if aggregate_monthly:
        # sum เดือน → ปี  (CO2_tonnes ต่อปี ต่อจังหวัด)
        df = (
            df.groupby(["province", "year"], as_index=False)["CO2_tonnes"]
            .sum()
        )
        df = df.sort_values(["province", "year"]).reset_index(drop=True)
        logger.info(f"รวมข้อมูล: {len(df)} แถว "
                    f"({df['province'].nunique()} จังหวัด, "
                    f"{df['year'].nunique()} ปี)")
    else:
        df = df.sort_values(["province", "year", "month"]).reset_index(drop=True)
        logger.info(f"รวมข้อมูล: {len(df)} แถว (รายเดือน)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Convenience: โหลดแล้วพร้อมส่งต่อ pipeline โดยตรง
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FOLDER = r"C:\Users\felm2\Desktop\sub\CO2_star_japan"


def load_odiac_for_pipeline(
    folder: str = DEFAULT_FOLDER,
    year_start: Optional[int] = None,
    year_end:   Optional[int] = None,
) -> pd.DataFrame:
    """
    โหลดข้อมูล ODIAC และคืน DataFrame รายปี-จังหวัด
    พร้อมใช้กับ run_pipeline() / run_gnn_pipeline() โดยตรง

    ตัวอย่าง:
        from odiac_loader import load_odiac_for_pipeline
        df = load_odiac_for_pipeline(year_start=2000, year_end=2022)
        result_df, next_pred, mape, r2, hist = run_gnn_pipeline(
            df=df,
            province_coords=THAILAND_PROVINCE_COORDS,
            n_years=1,
        )
    """
    return load_odiac_folder(
        folder=folder,
        year_start=year_start,
        year_end=year_end,
        aggregate_monthly=True,   # รายปี ตรงกับ pipeline
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI / Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    ys     = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ye     = int(sys.argv[3]) if len(sys.argv) > 3 else None

    df = load_odiac_for_pipeline(folder=folder, year_start=ys, year_end=ye)

    print("\n── ตัวอย่างข้อมูล 10 แถวแรก ──")
    print(df.head(10).to_string(index=False))
    print(f"\nshape  : {df.shape}")
    print(f"ปี     : {df['year'].min()} – {df['year'].max()}")
    print(f"จังหวัด: {df['province'].nunique()} จังหวัด")
    print(f"CO2 summary:\n{df['CO2_tonnes'].describe()}")