import os
import json
import base64
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Starship Coffee — Receipt OCR"
JSONL_PATH = os.path.join(os.path.dirname(__file__), "predictions.jsonl")

# Prefer Arvan envs, fallback to OpenAI-style envs
API_KEY = os.getenv("ARVAN_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("ARVAN_BASE_URL") or os.getenv("OPENAI_BASE_URL")

# Vision model name
DEFAULT_VISION_MODEL = (
    os.getenv("ARVAN_VISION_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4o-mini"
)

# Fail fast if missing
if not API_KEY:
    st.error("Missing API key. Set ARVAN_API_KEY (recommended) or OPENAI_API_KEY in your .env")
    st.stop()

# base_url can be None (SDK will use default). For Arvan it should be set.
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL if BASE_URL else None,
)

# -----------------------------
# Helpers
# -----------------------------
def _img_to_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _safe_json_loads(s: str):
    """Best-effort JSON parse."""
    s = (s or "").strip()
    if not s:
        return None
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None

def _normalize_output(obj: dict) -> dict:
    """Ensure the exact contract shape."""
    items = obj.get("items", []) if isinstance(obj, dict) else []
    total = obj.get("total", "0.00") if isinstance(obj, dict) else "0.00"

    cleaned_items = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        cleaned_items.append({
            "name": str(it.get("name", "")).strip()[:80],
            "qty": int(it.get("qty", 1) or 1),
            "unit_price": str(it.get("unit_price", "0.00")).strip(),
            "line_total": str(it.get("line_total", "0.00")).strip(),
        })

    return {"items": cleaned_items, "total": str(total).strip()}

def save_jsonl(record: dict, path: str = JSONL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# -----------------------------
# Vision call
# -----------------------------
def extract_receipt_structured(image_bytes: bytes, mime: str, model: str) -> dict:
    data_url = _img_to_data_url(image_bytes, mime)

    system_msg = (
        "You extract structured data from receipt images.\n"
        "Return ONLY valid JSON. No markdown. No extra keys.\n"
        "Output must match:\n"
        '{ "items":[{"name":"...","qty":1,"unit_price":"0.00","line_total":"0.00"}], "total":"0.00" }\n'
        "Rules:\n"
        "- Use simple item names.\n"
        "- qty must be an integer.\n"
        "- Prices must be strings like 12.34 (2 decimals when possible).\n"
        "- total must be the CURRENT payable total.\n"
        "- If multiple totals exist (e.g., crossed out vs current), choose the current one.\n"
        "- If you cannot read a field, still return valid JSON with best guess; do not add commentary.\n"
    )

    user_msg = [
        {"type": "text", "text": "Extract the receipt into the required JSON format."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content
    parsed = _safe_json_loads(content)
    if not parsed:
        return {"items": [], "total": "0.00"}

    return _normalize_output(parsed)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.sidebar.header("Settings")
model = st.sidebar.text_input("Vision model", value=DEFAULT_VISION_MODEL)

if BASE_URL:
    st.sidebar.caption(f"Base URL: {BASE_URL}")
else:
    st.sidebar.caption("Base URL: (default)")

uploaded = st.file_uploader("Upload a receipt image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded, caption="Uploaded receipt", use_container_width=True)

    with col2:
        with st.spinner("Reading receipt..."):
            out = extract_receipt_structured(
                image_bytes=uploaded.getvalue(),
                mime=uploaded.type or "image/jpeg",
                model=model,
            )

        st.subheader("Output JSON")
        st.code(json.dumps(out, indent=2), language="json")

        st.subheader("Items table")
        df = pd.DataFrame(out.get("items", [])) 
        if len(df) > 0:
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No items extracted.")

        st.subheader("Total")
        st.write(out.get("total", "0.00"))

        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "file_name": uploaded.name,
            "model": model,
            "output": out,
        }
        save_jsonl(record)

        st.success(f"Saved to {os.path.basename(JSONL_PATH)} ✅")
else:
    st.caption("Upload one receipt image to start.")
