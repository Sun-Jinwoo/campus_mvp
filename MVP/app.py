"""
Sistema de Validación Digital de Compras – Campus Inteligente
MVP – PDI 2026 | UAO  |  Modelo: YOLOv8
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import uuid
from datetime import datetime
import time

# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

MODEL_PATH           = "modelo.pt"
CONFIDENCE_THRESHOLD = 0.50
CLASS_NAMES = ["gaseosa", "gorra", "papas", "saco", "termo"]  # mismo orden que en el notebook

# True  → modelo de DETECCIÓN (bounding boxes, varias clases por imagen)
# False → modelo de CLASIFICACIÓN (una sola etiqueta por imagen)
IS_DETECTION_MODEL = True

PRICES = {
    "gaseosa": 3_500,
    "gorra":  25_000,
    "papas":   2_500,
    "saco":   45_000,
    "termo":  35_000,
}

EMOJIS = {
    "gaseosa": "🥤",
    "gorra":   "🧢",
    "papas":   "🍟",
    "saco":    "🧥",
    "termo":   "🫙",
}

# ══════════════════════════════════════════════════════════════
#  CARGA DEL MODELO YOLO
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando modelo YOLOv8…")
def load_model():
    import torch, tempfile, os

    # ── Intento 1: carga directa (modelo completo guardado con torch.save(model,...))
    try:
        m = YOLO(MODEL_PATH)
        return m, None
    except Exception:
        pass

    # ── Intento 2: state_dict puro con prefijo "model."
    #    (guardado como torch.save(model.state_dict(), path))
    try:
        raw_sd = torch.load(MODEL_PATH, map_location="cpu")

        if not isinstance(raw_sd, dict):
            raise ValueError("Formato desconocido")

        # Detectar si las claves tienen prefijo "model." y quitarlo
        first_key = next(iter(raw_sd))
        if first_key.startswith("model."):
            sd = {k[len("model."):]: v for k, v in raw_sd.items()}
        else:
            sd = raw_sd

        # Detectar número de clases mirando la capa final de clasificación
        # Clave esperada: "model.22.cv3.0.2.weight"  shape = [nc, C, 1, 1]
        nc = len(CLASS_NAMES)   # fallback al valor configurado
        for k, v in sd.items():
            if "cv3" in k and k.endswith(".2.weight") and v.ndim == 4:
                nc = v.shape[0]
                break

        # Construir YOLOv8n desde yaml con el nc correcto y cargar pesos
        yolo_wrapper = YOLO("yolov8n.yaml")
        yolo_wrapper.model.nc = nc

        missing, unexpected = yolo_wrapper.model.load_state_dict(sd, strict=False)
        yolo_wrapper.model.eval()
        return yolo_wrapper, None

    except Exception as e2:
        return None, str(e2)


# ══════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO (OpenCV – CLAHE)
# ══════════════════════════════════════════════════════════════

def preprocess_opencv(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ══════════════════════════════════════════════════════════════
#  INFERENCIA YOLO
# ══════════════════════════════════════════════════════════════

def classify_yolo(model, pil_img):
    processed = preprocess_opencv(pil_img)
    results   = model.predict(source=processed, verbose=False, conf=CONFIDENCE_THRESHOLD)
    detections = []
    for r in results:
        if IS_DETECTION_MODEL:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = model.names[cls]
                    detections.append((label, conf))
        else:
            if r.probs is not None:
                top1_idx  = int(r.probs.top1)
                top1_conf = float(r.probs.top1conf)
                label     = model.names[top1_idx]
                detections.append((label, top1_conf))
    return detections


def generate_ticket_code():
    now = datetime.now()
    uid = uuid.uuid4().hex[:8].upper()
    return f"UAO-{now.strftime('%Y%m%d-%H%M%S')}-{uid}"


# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.uao-header { background: linear-gradient(135deg, #8B0000 0%, #C0392B 100%); color: white; padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem; }
.uao-header h1 { margin: 0; font-size: 1.4rem; font-weight: 700; }
.uao-header p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
.step-badge { background: #f0f0f0; color: #333; font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; display: inline-block; }
.product-card { background: white; border: 1.5px solid #eee; border-radius: 14px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
.conf-high { color: #2e7d32; font-weight: 600; }
.conf-low  { color: #e65100; font-weight: 600; }
.ticket { background: white; border: 2px dashed #ccc; border-radius: 16px; padding: 1.8rem; font-family: 'Space Mono', monospace; font-size: 0.82rem; }
.ticket-header { text-align: center; border-bottom: 1px dashed #ccc; padding-bottom: 1rem; margin-bottom: 1rem; }
.ticket-title  { font-size: 1rem; font-weight: 700; letter-spacing: 1px; }
.ticket-row    { display: flex; justify-content: space-between; padding: 4px 0; }
.ticket-total  { border-top: 1px dashed #ccc; margin-top: 0.8rem; padding-top: 0.8rem; font-weight: 700; font-size: 0.95rem; }
.status-ok   { background:#e8f5e9; border:2px solid #2e7d32; color:#1b5e20; border-radius:12px; padding:1rem 1.5rem; text-align:center; font-size:1.3rem; font-weight:700; }
.status-fail { background:#ffebee; border:2px solid #c62828; color:#b71c1c; border-radius:12px; padding:1rem 1.5rem; text-align:center; font-size:1.3rem; font-weight:700; }
.code-box { background:#0d0d0d; color:#00e5ff; font-family:'Space Mono',monospace; font-size:1.1rem; letter-spacing:3px; padding:1rem; border-radius:10px; text-align:center; margin-top:0.6rem; }
.info-box { background:#fff8e1; border-left:4px solid #f9a825; padding:0.8rem 1rem; border-radius:0 8px 8px 0; font-size:0.85rem; color:#555; }
</style>
"""

# ══════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Validación de Compras – UAO", page_icon="🛒", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="uao-header">
    <div style="font-size:2.2rem">🛒</div>
    <div>
        <h1>Validación Digital de Compras</h1>
        <p>Campus Inteligente · PDI 2026 · Universidad Autónoma de Occidente</p>
    </div>
</div>
""", unsafe_allow_html=True)

model, load_error = load_model()

if load_error:
    st.error(f"No se pudo cargar **{MODEL_PATH}**: `{load_error}`")
    st.markdown("""<div class="info-box">
    Asegúrate de que <b>modelo.pt</b> esté en la misma carpeta que <b>app.py</b>.
    </div>""", unsafe_allow_html=True)
    st.stop()

st.success("✅ Modelo YOLOv8 cargado correctamente", icon="🤖")
st.divider()

# Paso 1
st.markdown('<span class="step-badge">Paso 1</span>', unsafe_allow_html=True)
st.subheader("📸 Captura de productos")
st.caption("Sube una o varias imágenes de los productos adquiridos.")

uploaded_files = st.file_uploader(
    "Arrastra o selecciona las imágenes",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("""<div class="info-box">👆 Sube las fotos para iniciar la validación.</div>""", unsafe_allow_html=True)
    st.stop()

# Paso 2
st.divider()
st.markdown('<span class="step-badge">Paso 2</span>', unsafe_allow_html=True)
st.subheader("🔍 Clasificación de productos")

all_detections = []
progress = st.progress(0, text="Analizando imágenes…")

for i, file in enumerate(uploaded_files):
    img  = Image.open(file).convert("RGB")
    dets = classify_yolo(model, img)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, use_column_width=True)
    with col2:
        if dets:
            for label, conf in dets:
                all_detections.append((label, conf))
                conf_class = "conf-high" if conf >= CONFIDENCE_THRESHOLD else "conf-low"
                emoji = EMOJIS.get(label, "📦")
                st.markdown(f"""
                <div class="product-card">
                    {emoji} <b>{label.capitalize()}</b>
                    &nbsp;·&nbsp;<span class="{conf_class}">{conf*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No se detectó ningún producto en esta imagen.")

    progress.progress((i + 1) / len(uploaded_files), text=f"Procesando {i+1}/{len(uploaded_files)}…")
    time.sleep(0.1)

progress.empty()

# Paso 3
st.divider()
st.markdown('<span class="step-badge">Paso 3</span>', unsafe_allow_html=True)
st.subheader("🧾 Ticket digital y validación")

valid_items = [(lbl, conf) for lbl, conf in all_detections if conf >= CONFIDENCE_THRESHOLD]

if valid_items:
    now         = datetime.now()
    ticket_code = generate_ticket_code()
    total       = sum(PRICES.get(lbl, 0) for lbl, _ in valid_items)

    rows_html = "".join(
        f'<div class="ticket-row"><span>{EMOJIS.get(lbl,"📦")} {lbl.capitalize()}</span><span>${PRICES.get(lbl,0):,}</span></div>'
        for lbl, _ in valid_items
    )

    st.markdown(f"""
    <div class="ticket">
        <div class="ticket-header">
            <div class="ticket-title">🛒 TICKET DE COMPRA</div>
            <div>{now.strftime('%d/%m/%Y  %H:%M:%S')}</div>
            <div style="color:#888;font-size:0.75rem">Campus UAO · Cafetería</div>
        </div>
        {rows_html}
        <div class="ticket-row ticket-total"><span>TOTAL</span><span>${total:,} COP</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="status-ok">✅ COMPRA VALIDADA</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Código único de entrega**")
    st.markdown(f'<div class="code-box">{ticket_code}</div>', unsafe_allow_html=True)
    st.caption("Presenta este código en caja para recibir tus productos.")

else:
    st.markdown('<div class="status-fail">❌ COMPRA NO VALIDADA</div>', unsafe_allow_html=True)
    st.error("No se detectaron productos con suficiente confianza. Prueba con imágenes más claras.")