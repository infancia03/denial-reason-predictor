import io, re
import numpy as np
import pandas as pd

EXPECTED_CANONICAL = ["cpt_code", "insurance_company", "physician", "payment", "balance", "denial_reason"]

KEYWORDS = [
    "cpt", "cpt code", "insurance", "insurance company", "payer",
    "physician", "physician name", "provider", "doctor",
    "payment", "payment amount", "paid", "balance",
    "denial", "denial reason", "denial code", "reason"
]

def _canonize(col: str) -> str:
    c = str(col).strip().lower()
    c = re.sub(r"[\n\r\t]", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    variants = {
        "cpt": "cpt_code", "cpt code": "cpt_code", "cptcode": "cpt_code",
        "insurance": "insurance_company", "insurance company": "insurance_company", "payer": "insurance_company",
        "physician": "physician", "physician name": "physician", "provider": "physician", "doctor": "physician",
        "payment": "payment", "payment amount": "payment", "amount paid": "payment", "paid": "payment",
        "balance": "balance", "balance amount": "balance",
        "denial": "denial_reason", "denial reason": "denial_reason",
        "denial code": "denial_reason", "reason": "denial_reason",
    }
    return variants.get(c, c)

def detect_header_row(content: bytes, is_excel: bool) -> int | None:
    try:
        preview = pd.read_excel(io.BytesIO(content), header=None, nrows=10, dtype=str) if is_excel \
                  else pd.read_csv(io.BytesIO(content), header=None, nrows=10, dtype=str, engine="python")
    except Exception:
        return None
    for i in range(min(10, len(preview))):
        row = preview.iloc[i].astype(str).fillna("").tolist()
        hits = sum(any(kw in cell.lower() for kw in KEYWORDS) for cell in row)
        if hits >= 2:
            return i
    return None

def currency_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = str(x).replace("$", "").replace(",", "").replace("(", "-").replace(")", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    try: return float(s)
    except: return np.nan

def read_any_table(uploaded):
    name = uploaded.name.lower()
    is_excel = name.endswith(".xls") or name.endswith(".xlsx")
    uploaded.seek(0)
    content = uploaded.read()
    header_row = detect_header_row(content, is_excel)

    # --- Load file ---
    df = pd.read_excel(io.BytesIO(content), header=header_row or 0, dtype=str) if is_excel \
         else pd.read_csv(io.BytesIO(content), header=header_row or 0, dtype=str, engine="python")

    # --- Normalize headers first ---
    df = df.rename(columns={c: _canonize(c) for c in df.columns})

    # --- Ensure all canonical cols exist ---
    for need in EXPECTED_CANONICAL:
        if need not in df.columns:
            df[need] = np.nan

    # --- Clean numeric cols ---
    for c in ["payment", "balance"]:
        df[c] = df[c].apply(currency_to_float)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    return df[EXPECTED_CANONICAL].copy(), header_row
