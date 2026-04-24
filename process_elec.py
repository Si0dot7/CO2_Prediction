import pandas as pd
import glob
import os
import re

# ===== CONFIG =====
INPUT_FOLDER = r"C:\Users\felm2\Desktop\elec"   # โฟลเดอร์ไฟล์ .xlsx
OUTPUT_PATH = "./old_data/elec.csv"

# ===== Thai → English province name mapping (77 จังหวัด + alias) =====
PROVINCE_TH_TO_EN = {
    "กระบี่": "Krabi",
    "กรุงเทพมหานคร": "Bangkok",
    "กาญจนบุรี": "Kanchanaburi",
    "กาฬสินธุ์": "Kalasin",
    "กำแพงเพชร": "Kamphaeng Phet",
    "ขอนแก่น": "Khon Kaen",
    "จันทบุรี": "Chanthaburi",
    "จันทรบุรี": "Chanthaburi",       # alias (typo ในบางไฟล์)
    "ฉะเชิงเทรา": "Chachoengsao",
    "ชลบุรี": "Chonburi",
    "ชัยนาท": "Chainat",
    "ชัยภูมิ": "Chaiyaphum",
    "ชุมพร": "Chumphon",
    "ตรัง": "Trang",
    "ตราด": "Trat",
    "ตาก": "Tak",
    "นครนายก": "Nakhon Nayok",
    "นครปฐม": "Nakhon Pathom",
    "นครพนม": "Nakhon Phanom",
    "นครราชสีมา": "Nakhon Ratchasima",
    "นครศรีธรรมราช": "Nakhon Si Thammarat",
    "นครสวรรค์": "Nakhon Sawan",
    "นนทบุรี": "Nonthaburi",
    "นราธิวาส": "Narathiwat",
    "น่าน": "Nan",
    "บึงกาฬ": "Bueng Kan",
    "บุรีรัมย์": "Buriram",
    "ปทุมธานี": "Pathum Thani",
    "ประจวบคีรีขันธ์": "Prachuap Khiri Khan",
    "ปราจีนบุรี": "Prachin Buri",
    "ปัตตานี": "Pattani",
    "พระนครศรีอยุธยา": "Phra Nakhon Si Ayutthaya",
    "พะเยา": "Phayao",
    "พังงา": "Phang Nga",
    "พัทลุง": "Phatthalung",
    "พิจิตร": "Phichit",
    "พิษณุโลก": "Phitsanulok",
    "ภูเก็ต": "Phuket",
    "มหาสารคาม": "Maha Sarakham",
    "มุกดาหาร": "Mukdahan",
    "ยะลา": "Yala",
    "ยโสธร": "Yasothon",
    "ระนอง": "Ranong",
    "ระยอง": "Rayong",
    "ราชบุรี": "Ratchaburi",
    "ร้อยเอ็ด": "Roi Et",
    "ลพบุรี": "Lopburi",
    "ลำปาง": "Lampang",
    "ลำพูน": "Lamphun",
    "ศรีสะเกษ": "Sisaket",
    "สกลนคร": "Sakon Nakhon",
    "สงขลา": "Songkhla",
    "สตูล": "Satun",
    "สมุทรปราการ": "Samut Prakan",
    "สมุทรสงคราม": "Samut Songkhram",
    "สมุทรสาคร": "Samut Sakhon",
    "สระบุรี": "Saraburi",
    "สระแก้ว": "Sa Kaeo",
    "สิงห์บุรี": "Sing Buri",
    "สุพรรณบุรี": "Suphan Buri",
    "สุราษฎร์ธานี": "Surat Thani",
    "สุราษฎ์ธานี": "Surat Thani",    # alias (typo ในบางไฟล์)
    "สุรินทร์": "Surin",
    "สุโขทัย": "Sukhothai",
    "หนองคาย": "Nong Khai",
    "หนองบัวลำภู": "Nong Bua Lamphu",
    "อำนาจเจริญ": "Amnat Charoen",
    "อุดรธานี": "Udon Thani",
    "อุตรดิตถ์": "Uttaradit",
    "อุทัยธานี": "Uthai Thani",
    "อุบลราชธานี": "Ubon Ratchathani",
    "อ่างทอง": "Ang Thong",
    "เชียงราย": "Chiang Rai",
    "เชียงใหม่": "Chiang Mai",
    "เพชรบุรี": "Phetchaburi",
    "เพชรบูรณ์": "Phetchabun",
    "เลย": "Loei",
    "แพร่": "Phrae",
    "แม่ฮ่องสอน": "Mae Hong Son",
}

ELEC_FEAT_COLS = [
    "industrial_electricity",
    "residential_electricity",
    "public_electricity",
    "agriculture_electricity",
]


def load_elec_profile(elec_df: "pd.DataFrame" = None,
                      csv_path: str = OUTPUT_PATH,
                      year_start: int = 2018,
                      year_end: int = 2022) -> "pd.DataFrame":
    """
    โหลดและ aggregate ข้อมูลไฟฟ้าเป็น static profile ต่อจังหวัด

    Parameters
    ----------
    elec_df : pd.DataFrame, optional
        DataFrame ที่โหลดไว้แล้ว (ถ้าไม่ส่งมาจะโหลดจาก csv_path)
    csv_path : str
        path ของ elec.csv (ใช้เมื่อ elec_df=None)
    year_start / year_end : int
        ช่วงปีที่ใช้ aggregate (default 2018–2022)

    Returns
    -------
    pd.DataFrame  index = ชื่อจังหวัด (English), columns = ELEC_FEAT_COLS
    ค่าถูก min-max normalize แล้ว [0, 1]
    """
    import pandas as pd
    import numpy as np

    if elec_df is None:
        elec_df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df = elec_df.copy()

    # แปลงชื่อไทย → อังกฤษ
    df["province_en"] = df["ชื่อจังหวัด"].map(PROVINCE_TH_TO_EN)
    unmapped = df[df["province_en"].isna()]["ชื่อจังหวัด"].unique()
    if len(unmapped) > 0:
        import warnings
        warnings.warn(f"[load_elec_profile] ไม่พบ mapping สำหรับ: {unmapped.tolist()}")

    # กรองช่วงปี
    df = df[df["year"].between(year_start, year_end)]

    # fill public_electricity ที่อาจเป็น NaN
    for col in ELEC_FEAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    # aggregate เฉลี่ยต่อจังหวัด (EN)
    profile = (
        df.dropna(subset=["province_en"])
          .groupby("province_en")[ELEC_FEAT_COLS]
          .mean()
    )

    # min-max normalize แต่ละ feature
    for col in ELEC_FEAT_COLS:
        col_min = profile[col].min()
        col_max = profile[col].max()
        denom = col_max - col_min if col_max > col_min else 1.0
        profile[col] = (profile[col] - col_min) / denom

    return profile  # index = province EN name

# ===== clean function =====
def clean_numeric(series):
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("N/A", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
        .replace("", 0)
        .astype(float)
    )

# ===== extract year from filename =====
import re

def extract_year(filename):
    # ---- หา 4 หลักก่อน (แม่นกว่า) ----
    match_4 = re.search(r"(20(1[8-9]|2[0-2]))", filename)
    if match_4:
        return int(match_4.group(1))

    # ---- fallback หา 2 หลัก ----
    match_2 = re.search(r"(6[1-6])", filename)
    if match_2:
        year_2digit = int(match_2.group(1))
        return 1957 + year_2digit  # 61 -> 2018

    # ---- error ----
    raise ValueError(f"หา year ไม่เจอในชื่อไฟล์: {filename}")
# ===== load files (รันเฉพาะตอนเรียก script โดยตรง) =====
if __name__ == "__main__":
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.xlsx"))

    all_df = []

    for file in files:
        df = pd.read_excel(file)

        # ---- year ----
        year = extract_year(os.path.basename(file))
        df["year"] = year

        # ---- clean columns ----
        cols = [
            "บ้านอยู่อาศัย(kWh)",
            "กิจการขนาดเล็ก(kWh)",
            "กิจการขนาดกลาง(kWh)",
            "กิจการขนาดใหญ่(kWh)",
            "กิจการเฉพาะอย่าง(kWh)",
            "ไฟฟ้าสาธารณะ(kWh)",
            "ราชการ/รัฐวิสาหกิจ(kWh)",
            "การสูบน้ำ(kWh)"
        ]

        for col in cols:
            if col in df.columns:
                df[col] = clean_numeric(df[col])
            else:
                df[col] = 0

        # ===== feature engineering =====
        df["industrial_electricity"] = (
            df["กิจการขนาดเล็ก(kWh)"] +
            df["กิจการขนาดกลาง(kWh)"] +
            df["กิจการขนาดใหญ่(kWh)"] +
            df["กิจการเฉพาะอย่าง(kWh)"]
        )

        df["residential_electricity"] = df["บ้านอยู่อาศัย(kWh)"]

        df["public_electricity"] = (
            df["ไฟฟ้าสาธารณะ(kWh)"] +
            df["ราชการ/รัฐวิสาหกิจ(kWh)"]
        )

        df["agriculture_electricity"] = df["การสูบน้ำ(kWh)"]

        # ===== select columns =====
        df_final = df[[
            "ชื่อจังหวัด",
            "year",
            "industrial_electricity",
            "residential_electricity",
            "public_electricity",
            "agriculture_electricity"
        ]]

        all_df.append(df_final)

    # ===== combine =====
    final_df = pd.concat(all_df, ignore_index=True)

    # ===== sort =====
    final_df = final_df.sort_values(by=["year"])

    # ===== save =====
    os.makedirs("./old_data", exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved to:", OUTPUT_PATH)