import streamlit as st
import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import zipfile
import io
import warnings
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SAM + MiniViT — Grãos de Café", layout="wide")
st.title("☕ Segmentação + Classificação de Grãos")
st.markdown("**MobileSAM** segmenta · **MiniViT** classifica o estádio de cada grão")

# ── Cores por estádio (atualizadas para suas 5 classes) ───
CORES_ESTADIO = {
    "Verde":      (34,  158,  34),   # verde
    "Cana":       (20,  160, 180),   # amarelo-esverdeado
    "Cereja":     (40,   40, 200),   # vermelho
    "Seco":       (80,   50,  20),   # marrom escuro
    "Cerscospera":(128,  30, 128),   # roxo (doença)
}

# ── Cache dos modelos ─────────────────────────────────────
@st.cache_resource
def carregar_sam_normal():
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_batch=32,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        box_nms_thresh=0.30,
        min_mask_region_area=300,
        crop_n_layers=0,
    )

@st.cache_resource
def carregar_sam_verde():
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_batch=32,
        points_per_side=48,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.80,
        box_nms_thresh=0.25,
        min_mask_region_area=200,
        crop_n_layers=1,
        crop_overlap_ratio=0.5,
    )

@st.cache_resource
def carregar_minivit():
    # ── Atualize aqui para o nome da sua pasta do novo modelo ──
    caminho = "minivit-custom-graos21"

    processor = ViTImageProcessor(
        do_resize=True,
        size={"height": 64, "width": 64},
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    modelo = ViTForImageClassification.from_pretrained(
        caminho,
        ignore_mismatched_sizes=True,
    )
    modelo.eval()
    return processor, modelo

vit_processor, vit_model = carregar_minivit()


# ── Detecta tipo de foto automaticamente ─────────────────
def detectar_tipo_foto(imagem_bgr):
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    mask_fundo = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 40, 255]))
    mask_graos = cv2.bitwise_not(mask_fundo)
    pixels = hsv[mask_graos > 0]
    if len(pixels) == 0:
        return "normal"
    hue = pixels[:, 0]
    pct_verde = ((hue >= 35) & (hue <= 85)).sum() / len(hue)
    return "verde" if pct_verde > 0.35 else "normal"


# ── Classificação ─────────────────────────────────────────
def classificar_grao(recorte_bgr, processor, modelo):
    pil_img = Image.fromarray(cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB))
    inputs  = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        logits = modelo(**inputs).logits
    probs     = torch.softmax(logits, dim=-1)[0]
    idx       = probs.argmax().item()
    confianca = probs[idx].item()
    label = modelo.config.id2label.get(idx, str(idx))
    return label, round(confianca, 3)


# ── Funções SAM ───────────────────────────────────────────
def remover_fundo_branco(imagem_bgr):
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    mask_fundo = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 40, 255]))
    mask_graos = cv2.bitwise_not(mask_fundo)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_graos = cv2.morphologyEx(mask_graos, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_graos = cv2.morphologyEx(mask_graos, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask_graos

def nms_por_mascara(graos, iou_threshold=0.25):
    if not graos: return graos
    graos = sorted(graos, key=lambda g: g["area"], reverse=True)
    mantidos = []
    for g in graos:
        sobreposto = False
        gx, gy, gw, gh = [int(v) for v in g["bbox"]]
        for m in mantidos:
            mx, my, mw, mh = [int(v) for v in m["bbox"]]
            ix1, iy1 = max(gx, mx), max(gy, my)
            ix2, iy2 = min(gx+gw, mx+mw), min(gy+gh, my+mh)
            if ix1 >= ix2 or iy1 >= iy2: continue
            sg = g["segmentation"][iy1:iy2, ix1:ix2].astype(bool)
            sm = m["segmentation"][iy1:iy2, ix1:ix2].astype(bool)
            inter = np.logical_and(sg, sm).sum()
            uniao = np.logical_or(sg, sm).sum()
            if uniao == 0: continue
            if (inter/uniao) > iou_threshold or (inter/(sg.sum()+1e-6)) > 0.7:
                sobreposto = True
                break
        if not sobreposto: mantidos.append(g)
    return mantidos

def watershed_separar(mascara_bin, min_distancia=15):
    dist   = ndimage.distance_transform_edt(mascara_bin)
    coords = peak_local_max(dist, min_distance=min_distancia, labels=mascara_bin)
    if len(coords) <= 1: return [mascara_bin]
    markers = np.zeros_like(mascara_bin, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1): markers[r, c] = idx
    rotulos = watershed(-dist, markers, mask=mascara_bin)
    return [
        (rotulos == rid).astype(np.uint8) * 255
        for rid in range(1, rotulos.max() + 1)
        if (rotulos == rid).sum() >= 300
    ]

def e_grao_valido(m, mask_graos_hsv):
    area = m["area"]
    if not (300 <= area <= 40000): return False
    x, y, w, h = [int(v) for v in m["bbox"]]
    if h == 0 or not (0.4 <= w/h <= 2.2): return False
    seg_crop = m["segmentation"][y:y+h, x:x+w].astype(np.uint8)
    hsv_crop = (mask_graos_hsv[y:y+h, x:x+w] // 255).astype(np.uint8)
    inter    = cv2.bitwise_and(seg_crop, hsv_crop)
    seg_sum  = seg_crop.sum()
    return seg_sum > 0 and (inter.sum() / seg_sum) >= 0.45

def recortar_grao(imagem_bgr, segmentacao, bbox, padding=4):
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(0, x - padding);    y1 = max(0, y - padding)
    x2 = min(imagem_bgr.shape[1], x+w+padding)
    y2 = min(imagem_bgr.shape[0], y+h+padding)
    recorte   = imagem_bgr[y1:y2, x1:x2].copy()
    fundo     = np.ones_like(recorte) * 255
    mask_crop = segmentacao[y1:y2, x1:x2]
    resultado = np.where(mask_crop[:, :, None], recorte, fundo)
    return resultado.astype(np.uint8)


# ── Interface ─────────────────────────────────────────────
arquivo = st.file_uploader("Envie a foto dos grãos", type=["jpg","jpeg","png"])

if arquivo:
    bytes_data = arquivo.getvalue()
    img_array  = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img        = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    H, W = img.shape[:2]
    if max(H, W) > 800:
        escala = 800 / max(H, W)
        img = cv2.resize(img, (int(W*escala), int(H*escala)))

    # ── Detecta tipo e carrega SAM adequado ──────────────
    tipo_foto = detectar_tipo_foto(img)
    generator = carregar_sam_verde() if tipo_foto == "verde" else carregar_sam_normal()
    st.info(f"Modo detectado: **{tipo_foto}** — parâmetros otimizados ativos")

    # ── SAM segmenta ─────────────────────────────────────
    with st.spinner("SAM segmentando..."):
        mask_hsv     = remover_fundo_branco(img)
        mascaras_sam = generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        graos_brutos = []
        for m in mascaras_sam:
            if not e_grao_valido(m, mask_hsv): continue
            if m["area"] > 5000:
                x, y, w, h = [int(v) for v in m["bbox"]]
                sub_mask = (m["segmentation"][y:y+h, x:x+w] * 255).astype(np.uint8)
                for sub in watershed_separar(sub_mask):
                    conts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if conts:
                        bx, by, bw, bh = cv2.boundingRect(conts[0])
                        full_seg = np.zeros_like(m["segmentation"], dtype=bool)
                        full_seg[y:y+h, x:x+w] = sub > 0
                        graos_brutos.append({
                            "segmentation": full_seg,
                            "bbox": [x+bx, y+by, bw, bh],
                            "area": int(sub.sum()/255),
                            "origem": "ws",
                        })
            else:
                graos_brutos.append({
                    "segmentation": m["segmentation"],
                    "bbox": list(m["bbox"]),
                    "area": m["area"],
                    "origem": "sam",
                })

        graos_finais = nms_por_mascara(graos_brutos)

    st.info(f"SAM detectou {len(graos_finais)} grãos — classificando...")

    # ── MiniViT classifica cada grão ─────────────────────
    progresso = st.progress(0, text="Classificando com MiniViT...")
    contagem  = {}

    for i, g in enumerate(graos_finais):
        recorte        = recortar_grao(img, g["segmentation"], g["bbox"])
        label, conf    = classificar_grao(recorte, vit_processor, vit_model)
        g["label"]     = label
        g["confianca"] = conf
        g["recorte"]   = recorte
        contagem[label] = contagem.get(label, 0) + 1
        progresso.progress(
            (i+1) / len(graos_finais),
            text=f"Classificando {i+1}/{len(graos_finais)}...",
        )

    progresso.empty()
    total = len(graos_finais)

    # ── Debug colorido por estádio ────────────────────────
    debug = img.copy()
    for g in graos_finais:
        bx, by, bw, bh = [int(v) for v in g["bbox"]]
        cor = CORES_ESTADIO.get(g["label"], (100, 100, 100))
        cv2.rectangle(debug, (bx, by), (bx+bw, by+bh), cor, 2)
        txt = f"{g['label'][:3].upper()} {g['confianca']:.0%}"
        cv2.putText(debug, txt, (bx+2, by+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, cor, 1)

    # ── Métricas ──────────────────────────────────────────
    st.success(f"✓ {total} grãos classificados!")
    st.subheader("Porcentagem por estádio")

    # Ordena pela ordem natural das classes
    ORDEM_CLASSES = ["Verde", "Cana", "Cereja", "Seco", "Cerscospera"]
    classes_encontradas = [c for c in ORDEM_CLASSES if c in contagem]
    # Adiciona qualquer classe inesperada do modelo no final
    for c in contagem:
        if c not in classes_encontradas:
            classes_encontradas.append(c)

    cols = st.columns(len(classes_encontradas))
    for col, cls in zip(cols, classes_encontradas):
        pct = contagem[cls] / total * 100
        col.metric(label=cls, value=f"{pct:.1f}%", delta=f"{contagem[cls]} grãos")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Classificação por estádio")
        st.image(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Legenda
    legenda = " &nbsp;·&nbsp; ".join(
        f"<b style='color:rgb{tuple(reversed(v))}'>{k}</b>"
        for k, v in CORES_ESTADIO.items()
        if k in contagem
    )
    st.markdown(f"**Legenda:** {legenda}", unsafe_allow_html=True)

    # ── Recortes por estádio ──────────────────────────────
    st.subheader("Recortes por estádio")
    for cls in classes_encontradas:
        graos_cls = [g for g in graos_finais if g["label"] == cls]
        pct = len(graos_cls) / total * 100
        with st.expander(f"{cls} — {len(graos_cls)} grãos ({pct:.1f}%)"):
            n_mostrar = min(12, len(graos_cls))
            cols_rec  = st.columns(n_mostrar)
            for col, g in zip(cols_rec, graos_cls[:n_mostrar]):
                col.image(
                    cv2.cvtColor(g["recorte"], cv2.COLOR_BGR2RGB),
                    caption=f"{g['confianca']:.0%}",
                    use_container_width=True,
                )

    # ── Download ZIP organizado por estádio ──────────────
    buffer_zip = io.BytesIO()
    with zipfile.ZipFile(buffer_zip, "w") as zf:
        for i, g in enumerate(graos_finais):
            _, buf = cv2.imencode(".jpg", g["recorte"])
            nome   = f"{g['label']}/grao_{i:04d}_{g['confianca']:.0%}.jpg"
            zf.writestr(nome, buf.tobytes())

    st.download_button(
        label="📥 Baixar recortes organizados por estádio (.zip)",
        data=buffer_zip.getvalue(),
        file_name="graos_classificados.zip",
        mime="application/zip",
    )