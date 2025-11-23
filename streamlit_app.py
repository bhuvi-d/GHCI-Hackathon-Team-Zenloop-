import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ---------- BASIC CONFIG ----------
st.set_page_config(page_title="Zenloop Categorizer", layout="wide")

MODEL_DIR = "bert_finetuned"
CSV_PATH = "augmented_ready.csv"
TAXONOMY_PATH = "taxonomy.json"
BERT_CONF_THRESHOLD = 0.80
LLM_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---------- THEME / TOGGLES ----------
THEME_TOGGLE = st.sidebar.toggle("Light / Dark Mode", value=False)
USE_LLM_FALLBACK = st.sidebar.checkbox("Enable LLM fallback (slower)", value=True)

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
.stApp { background: radial-gradient(circle at top left, #fdfbff 0%, #f6f2fb 40%, #efe7ff 100%); color:#3a2d4f; }

.result-card {
    background: rgba(255,255,255,0.85);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 10px 25px rgba(103, 80, 164, 0.18);
    border: 1px solid rgba(150,130,180,0.18);
    animation: fadeIn 0.4s ease-in-out;
}

.hero {
    padding:18px;
    border-radius:14px;
    background: linear-gradient(120deg,#e5d4ff,#f6e9ff,#ffe9fb);
    display:flex;
    gap:12px;
    align-items:center;
    box-shadow: 0 12px 28px rgba(123, 97, 255, 0.25);
}

.hero-title {
    font-size: 30px;
    font-weight: 700;
}

.hero-subtitle {
    font-size: 14px;
    opacity: 0.85;
}

.badge {
    padding:8px 14px;
    border-radius:999px;
    background:#c8aaff;
    color:#fff;
    font-weight:700;
    font-size:14px;
}

.ghci-badge {
    background: linear-gradient(120deg,#ff9a9e,#fad0c4);
    color:#3b1432;
    animation: pulse 1.6s infinite;
}

.stButton>button {
    border-radius: 999px;
    padding: 0.45rem 1.4rem;
    border: none;
    background: linear-gradient(120deg,#7c3aed,#ff6fb5);
    color: white;
    font-weight: 600;
    box-shadow: 0 10px 20px rgba(124,58,237,0.35);
    transition: transform 0.08s ease-out, box-shadow 0.08s ease-out, filter 0.1s ease-out;
}

.stButton>button:hover {
    cursor:pointer;
    transform: translateY(-1px);
    box-shadow: 0 14px 26px rgba(124,58,237,0.45);
    filter: brightness(1.03);
}

@keyframes fadeIn {
    from {opacity:0; transform:translateY(6px);}
    to {opacity:1; transform:translateY(0);}
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,154,158,0.5); }
    70% { transform: scale(1.03); box-shadow: 0 0 0 8px rgba(255,154,158,0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,154,158,0); }
}
</style>
"""

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
.stApp {
    background: radial-gradient(circle at top left, #231833 0%, #151021 45%, #0b0716 100%);
    color:#e8e0f5;
}

.result-card {
    background: rgba(19,16,37,0.9);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.7);
    border: 1px solid rgba(255,255,255,0.06);
    animation: fadeIn 0.4s ease-in-out;
}

.hero {
    padding:18px;
    border-radius:14px;
    background: linear-gradient(120deg,rgba(160,130,255,0.32),rgba(210,160,255,0.18),rgba(255,146,210,0.20));
    display:flex;
    gap:12px;
    align-items:center;
    box-shadow: 0 14px 34px rgba(0,0,0,0.8);
}

.hero-title {
    font-size: 30px;
    font-weight: 700;
}

.hero-subtitle {
    font-size: 14px;
    opacity: 0.88;
}

.badge {
    padding:8px 14px;
    border-radius:999px;
    background:#a688ff;
    color:#1b1430;
    font-weight:700;
    font-size:14px;
}

.ghci-badge {
    background: linear-gradient(120deg,#ff9a9e,#fad0c4);
    color:#2b122e;
    animation: pulse 1.6s infinite;
}

.stButton>button {
    border-radius: 999px;
    padding: 0.45rem 1.4rem;
    border: none;
    background: linear-gradient(120deg,#8b5cf6,#ec4899);
    color: white;
    font-weight: 600;
    box-shadow: 0 10px 22px rgba(15,23,42,0.9);
    transition: transform 0.08s ease-out, box-shadow 0.08s ease-out, filter 0.1s ease-out;
}

.stButton>button:hover {
    cursor:pointer;
    transform: translateY(-1px);
    box-shadow: 0 16px 30px rgba(15,23,42,1);
    filter: brightness(1.04);
}

@keyframes fadeIn {
    from {opacity:0; transform:translateY(6px);}
    to {opacity:1; transform:translateY(0);}
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,154,158,0.45); }
    70% { transform: scale(1.04); box-shadow: 0 0 0 10px rgba(255,154,158,0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,154,158,0); }
}
</style>
"""

st.markdown(LIGHT_CSS if THEME_TOGGLE else DARK_CSS, unsafe_allow_html=True)

HEADER = """
<div class="hero">
  <div style="display:flex;flex-direction:column;">
    <div class="hero-title">
      Team Zenloop — Demo
    </div>
    <div class="hero-subtitle">
      AI-powered financial transaction classification
    </div>
  </div>
  <div style="margin-left:auto;display:flex;gap:10px;align-items:center;">
    <span class="badge">BERT classifier</span>
    <span class="badge">LLM fallback</span>
    <span class="badge ghci-badge">GHCI Hackathon</span>
  </div>
</div>
"""
st.markdown(HEADER, unsafe_allow_html=True)

# ---------- LOADERS ----------

@st.cache_resource
def load_bert(model_dir: str):
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "le": le, "device": device}

@st.cache_resource
def load_llm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(LLM_ID)

    try:
        m = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            dtype=dtype
        ).to(device)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(LLM_ID).to(device)

    m.eval()
    return {"tokenizer": tok, "model": m, "device": device}

# ---------- CORE HELPERS ----------

def bert_predict(text, pack):
    tok = pack["tokenizer"]
    model = pack["model"]
    le = pack["le"]
    device = pack["device"]

    enc = tok(
        str(text).lower().strip(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pid = int(np.argmax(probs))
    label = le.inverse_transform([pid])[0]
    conf = float(probs[pid])
    return label, conf, probs

def run_llm_for_category(text, bert_label, bert_conf, taxonomy):
    try:
        model_pack = load_llm()
    except Exception as e:
        return bert_label, f"LLM load failed, kept BERT label: {bert_label} ({e})"

    tok = model_pack["tokenizer"]
    m = model_pack["model"]
    device = model_pack["device"]

    cats = taxonomy.get("categories", [])
    cats_str = ", ".join(cats) if cats else ""

    prompt = f"""You are a banking transaction categorization assistant.

Categories: {cats_str}

Decide the BEST category for this transaction from the list.
You may keep BERT's category if it clearly fits.

Return ONLY valid JSON, no extra text:
{{
  "final_category": "<ONE category from the list>",
  "explanation": "<very short one-line reason>"
}}

Transaction: "{text}"
BERT_category: "{bert_label}"
BERT_confidence: {bert_conf:.3f}

JSON:
""".strip()

    inputs = tok(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            out_ids = m.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
    except Exception as e:
        return bert_label, f"LLM generation failed, kept BERT label '{bert_label}' ({e})"

    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    raw = tok.decode(gen_ids, skip_special_tokens=True).strip()

    final_cat = bert_label
    explanation = f"LLM fallback used, kept BERT category '{bert_label}'."

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start:end+1])
            cand = obj.get("final_category", "").strip()
            expl = obj.get("explanation", "").strip()
            if cand in cats:
                final_cat = cand
                explanation = expl or f"LLM chose '{cand}' based on transaction text."
    except Exception:
        pass

    return final_cat, explanation

# ---------- TAXONOMY ----------
taxonomy = {}
if os.path.exists(TAXONOMY_PATH):
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

# keep 📂 here as requested
st.sidebar.header("📂 Taxonomy")
st.sidebar.write(taxonomy.get("categories", []))

# ---------- MAIN UI ----------
col1, col2 = st.columns([2, 1])

with col2:
    # keep ⚙️ here as requested
    st.subheader("⚙️ Config")
    st.write(f"Model dir: `{MODEL_DIR}`")
    st.write(f"Sample CSV: `{CSV_PATH}`")
    st.caption(
        "Single input uses BERT first. "
        "LLM fallback only runs when confidence is low or label is 'Other' "
        "and the LLM toggle is enabled."
    )
    device_txt = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Backend device: `{device_txt}`")

with col1:
    st.subheader("Single Transaction")
    tx = st.text_area(
        "Enter transaction text",
        height=90,
        placeholder="e.g. NEFT/AMAZON MKT 129.00",
    )
    run = st.button("Predict")

pack = load_bert(MODEL_DIR)

if run:
    if not tx:
        st.error("Please enter text")
    else:
        label_bert, conf_bert, probs = bert_predict(tx, pack)
        final_label = label_bert
        final_conf = conf_bert
        source = "bert"
        explanation = f"BERT is confident ({conf_bert:.1%}) this is '{label_bert}'."

        should_call_llm = (
            USE_LLM_FALLBACK
            and ((conf_bert < BERT_CONF_THRESHOLD) or (label_bert == "Other"))
        )

        if should_call_llm:
            with st.spinner("Using LLM fallback (may be slower on first call)..."):
                llm_label, llm_expl = run_llm_for_category(
                    tx, label_bert, conf_bert, taxonomy
                )
            final_label = llm_label
            final_conf = conf_bert
            source = "llm_fallback"
            explanation = llm_expl or explanation

        st.markdown(
            f'<div class="result-card">'
            f'<div style="font-size:14px;opacity:0.7;margin-bottom:4px;">Final category</div>'
            f'<h3 style="margin:0 0 4px 0;">{final_label}'
            f'<span style="float:right;opacity:0.7;font-size:13px;">BERT conf: {conf_bert:.3f}</span>'
            f'</h3>'
            f'<div style="font-size:12px;opacity:0.75;">Source: {source}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if explanation:
            st.markdown(
                "<div style='margin-top:8px;font-size:13px;opacity:0.8;'>"
                f"<b>Explanation:</b> {explanation}</div>",
                unsafe_allow_html=True,
            )

        dfp = pd.DataFrame([probs], columns=list(pack["le"].classes_))
        st.write("Per-category probabilities (BERT):")
        st.bar_chart(dfp.T)

# ---------- BATCH (BERT ONLY) ----------
st.markdown("---")
st.subheader("Batch Prediction (BERT only)")

upload = st.file_uploader("Upload CSV", type=["csv"])
run_batch = st.button("Run batch prediction")

def auto_pick_text_column(df):
    if "CLEAN_TEXT" in df.columns:
        return df["CLEAN_TEXT"]
    if "RAW_TEXT" in df.columns:
        return df["RAW_TEXT"]
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols:
        return df[text_cols[0]]
    return df.iloc[:, 0]

if upload:
    df = pd.read_csv(upload)
    st.write("Preview:")
    st.dataframe(df.head())
    if run_batch:
        col = auto_pick_text_column(df)
        results = []
        with st.spinner("Running BERT on all rows... (no LLM for speed)"):
            for t in col.astype(str).tolist():
                lab, cf, _ = bert_predict(t, pack)
                results.append((t, lab, cf))
        rdf = pd.DataFrame(results, columns=["text", "pred_label", "pred_conf"])

        st.subheader("Batch prediction preview")
        st.dataframe(rdf.head())

        st.download_button(
            "Download predictions",
            rdf.to_csv(index=False),
            "preds.csv",
        )

        st.subheader("Category distribution (batch)")
        counts = rdf["pred_label"].value_counts()

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=None,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.axis("equal")
        ax.legend(
            wedges,
            counts.index,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        st.pyplot(fig)

if st.button("Load sample CSV"):
    if os.path.exists(CSV_PATH):
        sdf = pd.read_csv(CSV_PATH, nrows=50)
        st.dataframe(sdf.head())
    else:
        st.error("Sample CSV not found at path: " + CSV_PATH)
