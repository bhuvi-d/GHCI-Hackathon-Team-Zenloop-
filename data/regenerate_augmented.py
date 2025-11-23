#!/usr/bin/env python3
"""
regenerate_augmented.py

Input:
  CSV with columns like:
    trans_id, trans_date_trans_time, cc_num, merchant, category, amt, ...

Output:
  CSV with ALL original columns +
  ORIGINAL_ROW_ID, SOURCE, RAW_TEXT, SYN_RAW_TEXT, CLEAN_TEXT,
  CATEGORY, AMOUNT, IS_CREDIT, IS_DEBIT

We:
  - Build RAW_TEXT from merchant + amount
  - Generate N noisy SYN_RAW_TEXT variants
  - Create CLEAN_TEXT for modeling
  - Heuristically infer CATEGORY for bootstrapping
"""

import argparse
import pandas as pd
import numpy as np
import random
import re

random.seed(42)
np.random.seed(42)

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        s = str(x).replace(',', '').strip()
        if s == "":
            return None
        return float(s)
    except:
        return None

def detect_columns(sample_df):
    cols = list(sample_df.columns)
    merchant_col = None
    withdraw_cols = []
    deposit_cols = []

    for c in cols:
        lc = c.lower()
        if merchant_col is None and any(k in lc for k in ["merchant", "narr", "detail", "description", "partic"]):
            merchant_col = c
        if any(k in lc for k in ["withdraw", "debit"]):
            withdraw_cols.append(c)
        if any(k in lc for k in ["deposit", "credit"]):
            deposit_cols.append(c)
        # our case: "amt" → treat as debit amount
        if "amt" == lc or "amount" == lc:
            withdraw_cols.append(c)

    if merchant_col is None:
        # fallback: first object column
        for c in cols:
            if sample_df[c].dtype == object:
                merchant_col = c
                break

    return merchant_col, withdraw_cols, deposit_cols

def build_raw_text(merchant, amount):
    m = "" if pd.isna(merchant) else str(merchant).strip()
    if amount is None:
        return m
    try:
        return f"{m} {amount:.2f}".strip()
    except:
        return f"{m} {amount}".strip()

def introduce_noise(s):
    s = "" if pd.isna(s) else str(s).strip()
    # randomly drop vowels
    if random.random() < 0.18:
        s = re.sub(r'[aeiouAEIOU]', '', s)
    # random channel prefix
    if random.random() < 0.28:
        ch = random.choice(["UPI", "POS", "NEFT", "IMPS", "CARD", "ATM"])
        s = f"{ch}/{s}"
    # random marketplace suffix
    if random.random() < 0.22:
        suf = random.choice(["MKT","MKTPLC","MKT#"+str(random.randint(100,999)),"PAY","MALL"])
        s = f"{s} {suf}"
    # random internal separators
    if random.random() < 0.22:
        s = s.replace(" ", random.choice([" ", "*", "_", " / ", "-"]))
    # mask long digit groups
    s = re.sub(r'\d{4,}', lambda m: m.group(0)[:2] + "X"*(len(m.group(0))-4) + m.group(0)[-2:], s)
    return s

def clean_text(s):
    s = "" if pd.isna(s) else str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b\d+\b', ' ', s)   # remove standalone numbers
    s = re.sub(r'\s+', ' ', s).strip()
    return s

CATEGORY_KEYWORDS = {
    "Shopping":["amazon","flipkart","mkt","mktplc","shop","shopping","store","mall","paytm","myntra"],
    "Groceries":["grocery","bigbazaar","dmart","bigbasket","supermarket","reliancefresh"],
    "Dining":["starbucks","cafe","restaurant","dominos","pizza","kfc"],
    "Fuel":["shell","petrol","fuel","bharat","indianoil","bp"],
    "Utilities":["electric","electricity","water","bill","airtel","vodafone","broadband"],
    "Subscriptions":["spotify","netflix","prime","hotstar","zee5"],
    "Transfer":["transfer","neft","imps","rtgs","upi","trf"],
    "Salary":["salary","payroll","employer"],
    "Cash Withdrawal":["atm","withdraw","cash wdl","wdl"],
    "Bank Fee":["fee","charges","tds","service charge"]
}

def predict_category(clean):
    if not clean:
        return "Other"
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in clean:
                return cat
    if "payment" in clean:
        return "Shopping"
    return "Other"

def process_file(input_path, output_path, variants=3, chunksize=5000, cap=None):
    # auto-detect encoding + delimiter
    encodings = ['utf-8','latin1','cp1252','utf-16']
    sample = None
    encoding_used = None
    for enc in encodings:
        try:
            sample = pd.read_csv(input_path, nrows=20, sep=None, engine='python', encoding=enc, on_bad_lines='skip')
            encoding_used = enc
            break
        except Exception:
            continue
    if sample is None:
        sample = pd.read_csv(input_path, nrows=20, encoding='latin1', on_bad_lines='skip')
        encoding_used = 'latin1'

    merchant_col, withdraw_cols, deposit_cols = detect_columns(sample)
    print(f"Using encoding: {encoding_used}")
    print(f"Detected merchant column: {merchant_col}")
    print(f"Withdraw cols: {withdraw_cols}, deposit cols: {deposit_cols}")

    reader = pd.read_csv(input_path, chunksize=chunksize,
                         iterator=True, engine='python',
                         encoding=encoding_used, on_bad_lines='skip')

    first_write = True
    row_id = 0

    for chunk in reader:
        out_rows = []
        for _, row in chunk.iterrows():
            row_id += 1
            if cap and row_id > cap:
                break

            # detect amount: prefer withdraw, then deposit
            amt = None; is_credit=False; is_debit=False
            for c in withdraw_cols:
                if c in row and str(row[c]).strip() != "":
                    v = safe_float(row[c])
                    if v is not None and v > 0:
                        amt = v; is_debit = True; break
            if amt is None:
                for c in deposit_cols:
                    if c in row and str(row[c]).strip() != "":
                        v = safe_float(row[c])
                        if v is not None and v > 0:
                            amt = v; is_credit = True; break

            merchant_val = row[merchant_col] if merchant_col in row else ""
            raw_text = build_raw_text(merchant_val, amt)

            clean = clean_text(raw_text)
            cat = predict_category(clean)

            base = {
                "ORIGINAL_ROW_ID": row_id,
                "SOURCE": "original",
                **{c: row[c] for c in chunk.columns},
                "RAW_TEXT": raw_text,
                "SYN_RAW_TEXT": raw_text,
                "CLEAN_TEXT": clean,
                "CATEGORY": cat,
                "AMOUNT": amt,
                "IS_CREDIT": bool(is_credit),
                "IS_DEBIT": bool(is_debit),
            }
            out_rows.append(base)

            # synthetic variants
            for v in range(variants):
                syn = introduce_noise(raw_text)
                syn_clean = clean_text(syn)
                syn_cat = predict_category(syn_clean)
                aug = {
                    "ORIGINAL_ROW_ID": row_id,
                    "SOURCE": f"aug_v{v+1}",
                    **{c: row[c] for c in chunk.columns},
                    "RAW_TEXT": raw_text,
                    "SYN_RAW_TEXT": syn,
                    "CLEAN_TEXT": syn_clean,
                    "CATEGORY": syn_cat,
                    "AMOUNT": amt,
                    "IS_CREDIT": bool(is_credit),
                    "IS_DEBIT": bool(is_debit),
                }
                out_rows.append(aug)

        out_df = pd.DataFrame(out_rows)
        if first_write:
            out_df.to_csv(output_path, index=False, encoding='utf-8')
            first_write = False
        else:
            out_df.to_csv(output_path, index=False, header=False, mode='a', encoding='utf-8')

        if cap and row_id >= cap:
            break

    print(f"Done. Augmented file saved to: {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input","-i", required=True, help="Input CSV path (base data)")
    p.add_argument("--output","-o", default="augmented_ready.csv", help="Output CSV path")
    p.add_argument("--variants","-v", type=int, default=3, help="Synthetic variants per row")
    p.add_argument("--chunksize", type=int, default=5000, help="Chunk size for streaming")
    p.add_argument("--cap", type=int, default=None, help="Max rows to process (for testing)")
    args = p.parse_args()
    process_file(args.input, args.output,
                 variants=args.variants,
                 chunksize=args.chunksize,
                 cap=args.cap)
