"""
process_gpp.py
──────────────────────────────────────────────────────────────────────────────
Clean & parse GPP data จาก NESDC Excel (GPP-1995-2024.xlsx)
วิธีใช้:
  1. import ใน script อื่น  →  df = load_gpp(r"path/to/GPP-1995-2024.xlsx")
  2. รัน standalone         →  python process_gpp.py
──────────────────────────────────────────────────────────────────────────────
Output schema:
  province  | str  — ชื่อภาษาอังกฤษ (matched กับ ODIAC province list)
  year      | int  — 1995–2024
  sector    | str  — ชื่อ sector (ดู SECTORS_KEEP)
  value     | float — ล้านบาท (current prices); GPP per capita = บาท/คน
──────────────────────────────────────────────────────────────────────────────
"""
import re
import warnings
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0. ชื่อ sector ที่ต้องการ
# ─────────────────────────────────────────────────────────────────────────────
SECTORS_KEEP = [
    "Gross provincial product (GPP)",   # GPP รวม (ล้านบาท)
    "GPP Per capita (Baht)",            # GPP per capita
    "Population (1,000 persons)",       # ประชากร
    "Agriculture, forestry and fishing",
    "Manufacturing",
    "Transportation and storage",
    "Construction",
    "Electricity, gas, steam and air conditioning supply",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Province name mapping  GPP (NESDC) → ODIAC/GNN
# ─────────────────────────────────────────────────────────────────────────────
GPP_TO_ODIAC: dict[str, str] = {
    # ── ชื่อที่ตรงกันสมบูรณ์ (title case) ──────────────────────────────────
    "Khon Kaen":             "Khon Kaen",
    "Udon Thani":            "Udon Thani",
    "Loei":                  "Loei",
    "Nong Khai":             "Nong Khai",
    "Maha Sarakham":         "Maha Sarakham",
    "Roi Et":                "Roi Et",
    "Kalasin":               "Kalasin",
    "Sakon Nakhon":          "Sakon Nakhon",
    "Nakhon Phanom":         "Nakhon Phanom",
    "Yasothon":              "Yasothon",
    "Chaiyaphum":            "Chaiyaphum",
    "Nong Bua Lamphu":       "Nong Bua Lamphu",
    "Bueng Kan":             "Bueng Kan",
    "Ubon Ratchathani":      "Ubon Ratchathani",
    "Surin":                 "Surin",
    "Chiang Mai":            "Chiang Mai",
    "Lampang":               "Lampang",
    "Lamphun":               "Lamphun",
    "Mae Hong Son":          "Mae Hong Son",
    "Chiang Rai":            "Chiang Rai",
    "Phayao":                "Phayao",
    "Phrae":                 "Phrae",
    "Nan":                   "Nan",
    "Uttaradit":             "Uttaradit",
    "Tak":                   "Tak",
    "Sukhothai":             "Sukhothai",
    "Phitsanulok":           "Phitsanulok",
    "Phichit":               "Phichit",
    "Phetchabun":            "Phetchabun",
    "Nakhon Sawan":          "Nakhon Sawan",
    "Uthai Thani":           "Uthai Thani",
    "Ang Thong":             "Ang Thong",
    "Saraburi":              "Saraburi",
    "Kanchanaburi":          "Kanchanaburi",
    "Ratchaburi":            "Ratchaburi",
    "Samut Songkhram":       "Samut Songkhram",
    "Nonthaburi":            "Nonthaburi",
    "Pathum Thani":          "Pathum Thani",
    "Samut Prakan":          "Samut Prakan",
    "Rayong":                "Rayong",
    "Chanthaburi":           "Chanthaburi",
    "Trat":                  "Trat",
    "Nakhon Nayok":          "Nakhon Nayok",
    "Chachoengsao":          "Chachoengsao",
    "Surat Thani":           "Surat Thani",
    "Ranong":                "Ranong",
    "Chumphon":              "Chumphon",
    "Phuket":                "Phuket",
    "Krabi":                 "Krabi",
    "Phatthalung":           "Phatthalung",
    "Trang":                 "Trang",
    "Satun":                 "Satun",
    "Songkhla":              "Songkhla",
    "Pattani":               "Pattani",
    "Yala":                  "Yala",
    "Narathiwat":            "Narathiwat",
    # ── ชื่อที่ต่างกัน (GPP raw title → ODIAC) ──────────────────────────────
    "Amnat Chareon":            "Amnat Charoen",
    "Bangkok Metropolis":       "Bangkok",
    "Buri Ram":                 "Buriram",
    "Kam Phaeng Phet":          "Kamphaeng Phet",
    "Mukdahan":                 "Mukdahan",
    "Nongbua Lamphu":           "Nong Bua Lamphu",
    "Phachuap Khiri Khan":      "Prachuap Khiri Khan",
    "Phangnga":                 "Phang Nga",
    "Phra Nakhon Sri Ayuthaya": "Phra Nakhon Si Ayutthaya",
    "Sa Kaew":                  "Sa Kaeo",
    "Si Sa Ket":                "Sisaket",
    "Singburi":                 "Sing Buri",
    "Chai Nat":                 "Chainat",
    "Chon Buri":                "Chonburi",
    "Lop Buri":                 "Lopburi",
    "Nakhon Pathom":            "Nakhon Pathom",
    "Nakhon Ratchasima":        "Nakhon Ratchasima",
    "Nakhon Si Thammarat":      "Nakhon Si Thammarat",
    "Phang Nga":                "Phang Nga",
    "Prachin Buri":             "Prachin Buri",
    "Prachuap Khiri Khan":      "Prachuap Khiri Khan",
    "Samut Sakhon":             "Samut Sakhon",
    "Suphan Buri":              "Suphan Buri",
    "Phetchaburi":              "Phetchaburi",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: clean numeric string → float
# ─────────────────────────────────────────────────────────────────────────────
def _to_float(v) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except (ValueError, TypeError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper: clean year value → int
#    รองรับทั้ง float (1995.0), string suffix ("2020r", "2024p")
# ─────────────────────────────────────────────────────────────────────────────
def _clean_year(v) -> int | None:
    # float เช่น 1995.0 → แปลงเป็น int ก่อนเพื่อหลีกเลี่ยง "19950" จาก "1995.0"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        v = int(v)
    s = re.sub(r"[^\d]", "", str(v))   # ตัดทุกอย่างที่ไม่ใช่ตัวเลข
    return int(s) if re.match(r"^(19|20)\d{2}$", s) else None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Core parser
# ─────────────────────────────────────────────────────────────────────────────
def _parse_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    รับ DataFrame ดิบจาก pd.read_excel (header=None)
    โครงสร้างไฟล์ GPP-1995-2024.xlsx:
      col 0 = label (ชื่อจังหวัด / ชื่อ sector)
      col 1+ = ข้อมูลปี 1995–2024

    คืน long-format DataFrame ที่ clean แล้ว
    """
    df = df_raw.reset_index(drop=True).copy()

    # label อยู่ col 0, ข้อมูลเริ่มจาก col 1
    label_col = 0
    data_start = 1

    # ── หา province block starts ─────────────────────────────────────────────
    # pattern: "0101 - KHON KAEN" หรือ "0601 - SARABURI" เป็นต้น
    prov_pattern = re.compile(r"^\d{4}\s*-\s*(.+)$")
    block_starts: list[tuple[int, str]] = []

    for idx in range(len(df)):
        cell = str(df.iloc[idx, label_col]).strip()
        m = prov_pattern.match(cell)
        if m:
            block_starts.append((idx, m.group(1).strip().title()))

    records = []

    for bi, (start_row, raw_name) in enumerate(block_starts):
        # ── skip Chain Volume blocks ─────────────────────────────────────────
        # มองย้อนกลับไปหา marker "CHAIN" หรือ "REFERENCE YEAR"
        is_chain = False
        for look_back in range(1, 8):
            prev_idx = start_row - look_back
            if prev_idx < 0:
                break
            prev_cell = str(df.iloc[prev_idx, label_col]).upper()
            if "CHAIN" in prev_cell or "REFERENCE YEAR" in prev_cell:
                is_chain = True
                break
            if "CURRENT" in prev_cell or "GROSS PROVINCIAL" in prev_cell:
                break
        if is_chain:
            continue

        # ── map province name ────────────────────────────────────────────────
        odiac_name = GPP_TO_ODIAC.get(raw_name)
        if odiac_name is None:
            # fallback: case-insensitive match
            for k, v in GPP_TO_ODIAC.items():
                if k.lower() == raw_name.lower():
                    odiac_name = v
                    break
        if odiac_name is None:
            warnings.warn(f"[process_gpp] ไม่พบ mapping: '{raw_name}' — ข้ามจังหวัดนี้")
            continue

        # ── หา year row ──────────────────────────────────────────────────────
        end_row = block_starts[bi + 1][0] if bi + 1 < len(block_starts) else start_row + 40
        years: list[int] = []
        year_col_start: int | None = None

        for ri in range(start_row, min(end_row, start_row + 10)):
            row_vals = df.iloc[ri, data_start:].tolist()
            candidate = [_clean_year(v) for v in row_vals]
            valid = [(data_start + i, y) for i, y in enumerate(candidate) if y is not None]
            if len(valid) >= 10:
                year_col_start = valid[0][0]
                years = [y for _, y in valid]
                break

        if not years:
            warnings.warn(f"[process_gpp] หา year row ไม่เจอสำหรับ: {raw_name}")
            continue

        # ── อ่าน sector rows ─────────────────────────────────────────────────
        for ri in range(start_row, min(end_row, start_row + 40)):
            sector_cell = str(df.iloc[ri, label_col]).strip()
            sector_clean = " ".join(sector_cell.split())   # normalize whitespace/newlines

            if sector_clean not in SECTORS_KEEP:
                continue

            data_vals = df.iloc[ri, year_col_start: year_col_start + len(years)].tolist()

            for yr, val in zip(years, data_vals):
                records.append({
                    "province": odiac_name,
                    "year":     yr,
                    "sector":   sector_clean,
                    "value":    _to_float(val),
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Public API: load_gpp()
# ─────────────────────────────────────────────────────────────────────────────
def load_gpp(
    path: str,
    sheets: list[str] | None = None,
    year_start: int = 2000,
    year_end:   int = 2024,
) -> pd.DataFrame:
    """
    โหลด GPP จาก Excel และ return long-format DataFrame ที่ clean แล้ว

    Parameters
    ----------
    path       : path ของ .xlsx
    sheets     : รายชื่อ sheet (default = ทุก region sheet)
    year_start : กรองปีเริ่มต้น
    year_end   : กรองปีสุดท้าย

    Returns
    -------
    pd.DataFrame  columns = [province, year, sector, value]
    """
    if sheets is None:
        sheets = ["NE", "NO", "SO", "EA", "WE", "CE", "BKK&VIC"]

    raw_sheets = pd.read_excel(path, sheet_name=sheets, header=None)
    df_raw = pd.concat(raw_sheets.values(), ignore_index=True)

    df = _parse_raw(df_raw)

    # ── กรองปีและ drop NaN ──────────────────────────────────────────────────
    df = df[df["year"].between(year_start, year_end)].copy()
    df = df.dropna(subset=["value"])
    df = df.sort_values(["province", "sector", "year"]).reset_index(drop=True)

    n_prov = df["province"].nunique()
    print(f"[load_gpp] OK → {len(df):,} rows | {n_prov} provinces | years {df['year'].min()}–{df['year'].max()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pivot helper: wide format สำหรับ merge กับ ODIAC
# ─────────────────────────────────────────────────────────────────────────────
def pivot_gpp(df: pd.DataFrame) -> pd.DataFrame:
    """
    แปลง long → wide format
    columns: province, year, gpp, gpp_per_capita, population,
             manufacturing, transport, agriculture, construction, electricity

    ใช้ merge กับ ODIAC df ได้เลย:
        odiac.merge(pivot_gpp(gpp_df), on=["province","year"], how="left")
    """
    RENAME = {
        "Gross provincial product (GPP)":            "gpp",
        "GPP Per capita (Baht)":                     "gpp_per_capita",
        "Population (1,000 persons)":                "population",
        "Agriculture, forestry and fishing":         "gpp_agriculture",
        "Manufacturing":                             "gpp_manufacturing",
        "Transportation and storage":                "gpp_transport",
        "Construction":                              "gpp_construction",
        "Electricity, gas, steam and air conditioning supply": "gpp_electricity",
    }
    df2 = df.copy()
    df2["sector"] = df2["sector"].map(RENAME).fillna(df2["sector"])

    wide = df2.pivot_table(
        index=["province", "year"],
        columns="sector",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # ── derived features ────────────────────────────────────────────────────
    if "gpp" in wide.columns and "gpp_manufacturing" in wide.columns:
        wide["manufacturing_share"] = wide["gpp_manufacturing"] / wide["gpp"]
    if "gpp" in wide.columns and "gpp_transport" in wide.columns:
        wide["transport_share_gpp"] = wide["gpp_transport"] / wide["gpp"]
    if "gpp" in wide.columns and "gpp_agriculture" in wide.columns:
        wide["agriculture_share"] = wide["gpp_agriculture"] / wide["gpp"]

    # YoY growth ต่อจังหวัด (leading indicator)
    wide = wide.sort_values(["province", "year"])
    wide["gpp_growth"] = (
        wide.groupby("province")["gpp"]
            .pct_change()
            .clip(-0.5, 0.5)
    )

    return wide


# ─────────────────────────────────────────────────────────────────────────────
# 7. Standalone run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os

    EXCEL_PATH = r"C:\Users\felm2\Downloads\GPP-1995-2024.xlsx"
    OUT_LONG   = "./old_data/gpp_long.csv"
    OUT_WIDE   = "./old_data/gpp_wide.csv"

    if not os.path.exists(EXCEL_PATH):
        print(f"ไม่พบไฟล์: {EXCEL_PATH}")
        sys.exit(1)

    os.makedirs("./old_data", exist_ok=True)

    # ── long format ──────────────────────────────────────────────────────────
    df_long = load_gpp(EXCEL_PATH)
    df_long.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
    print(f"Saved long → {OUT_LONG}")

    # ── wide format ──────────────────────────────────────────────────────────
    df_wide = pivot_gpp(df_long)
    df_wide.to_csv(OUT_WIDE, index=False, encoding="utf-8-sig")
    print(f"Saved wide → {OUT_WIDE}")

    # ── quick check ──────────────────────────────────────────────────────────
    print("\n── Wide sample (Bangkok 2019–2022) ──")
    bkk = df_wide[
        (df_wide["province"] == "Bangkok") &
        (df_wide["year"].between(2019, 2022))
    ]
    print(bkk.to_string(index=False))

    print("\n── Missing provinces (ถ้ามี) ──")
    TARGET_77 = {v for v in GPP_TO_ODIAC.values()}
    found = set(df_wide["province"].unique())
    missing = TARGET_77 - found
    if missing:
        print("Missing:", sorted(missing))
    else:
        print("ครบ 77 จังหวัด ✓")