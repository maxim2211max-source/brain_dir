# app_demo.py – демо / минимальный вариант – 60 строк
import streamlit as st, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import sys, json
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from CV_fd_deploy.utils.face_blur import blur_faces  # Правильный импорт функции




ROOT = Path(__file__).parent
MODEL = ROOT / "/home/adminadmin/brain_dir/CV_fd_deploy/models/best_fd.pt"
RESULT = ROOT / "/home/adminadmin/brain_dir/CV_fd_deploy/assets/results.csv"     # ваш единственный CSV


st.set_page_config(page_title="Face Blur App", page_icon="👤", layout="wide")
st.title("Детекция и размытие лиц")

@st.cache_resource
def load_model():
    return YOLO(str(MODEL))

model = load_model()

# ---------- 1. загрузка ----------
uploaded = st.file_uploader("Выберите изображения", type=["png","jpg","jpeg","bmp","webp"],
                            accept_multiple_files=True)
if not uploaded:
    st.info("👈 Загрузите хотя бы одно изображение")
    st.stop()

# ---------- 2. две колонки ----------
for file in uploaded:
    img = Image.open(file).convert("RGB")
    c1, c2 = st.columns(2)

    # исходник
    with c1:
        st.image(img, use_column_width=True)

    # детект + blur
    with c2:
        arr = np.asarray(img).copy()
        results = model(arr)
        n = 0
        for r in results:
            for b in r.boxes:
                conf = float(b.conf)
                if conf < .5:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(arr, f"{conf:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                n += 1
        blurred, _ = blur_faces(arr, model, blur_strength=51)
        st.image(Image.fromarray(blurred), use_column_width=True)
        st.caption(f"лиц: **{n}**")



st.title("Анализ обучения модели")

if RESULT.exists():
    df = pd.read_csv(RESULT)
    df.columns = df.columns.str.replace("/", "_")
    st.success(f"✅ Загружено {len(df)} эпох")
else:
    st.error(f"❌ Файл не найден: {RESULT}")
    st.stop()

# ---------- 1. mAP50 ----------
fig1, ax1 = plt.subplots(figsize=(6, 3))
ax1.plot(df["epoch"], df["metrics_mAP50(B)"], "b-o", lw=2)
best_idx = df["metrics_mAP50(B)"].idxmax()
ax1.plot(df.loc[best_idx, "epoch"], df.loc[best_idx, "metrics_mAP50(B)"], "ro", ms=8)
ax1.set(xlabel="Эпоха", ylabel="mAP50", title="Training Progress (mAP50)")
ax1.grid(alpha=.3); ax1.set_ylim(0.6, 1)

# ---------- 2. PR-curve ----------
p_col = "metrics_precision(B)"
r_col = "metrics_recall(B)"
if p_col in df.columns and r_col in df.columns:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    pr_df = df[[r_col, p_col]].sort_values(r_col)
    ax2.plot(pr_df[r_col], pr_df[p_col], drawstyle='steps-post', color='tab:blue', lw=2)
    ax2.set(xlabel="Recall", ylabel="Precision", title="PR-curve")
    ax2.grid(alpha=.3); ax2.set_ylim(0.8, 1.02)

# ---------- вывод в две колонки ----------
c1, c2 = st.columns(2)
c1.pyplot(fig1)
c2.pyplot(fig2)

# ---------- статистика ----------
col1 = st.columns(1)
st.metric("Лучший mAP50", f"{df['metrics_mAP50(B)'].max():.3f}")


# ---------- 3. Потери (loss) ----------
st.markdown("---")
st.subheader("Динамика потерь")

# график 1: cls-loss
fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.plot(df["epoch"], df["train_cls_loss"], label="train_cls", marker="o")
if "val_cls_loss" in df.columns:
    ax3.plot(df["epoch"], df["val_cls_loss"], label="val_cls", marker="s")
ax3.set(xlabel="Эпоха", ylabel="cls loss", title="Classification loss")
ax3.legend(); ax3.grid(alpha=.3)


# график 2: box-loss
fig4, ax4 = plt.subplots(figsize=(6, 3))
ax4.plot(df["epoch"], df["train_box_loss"], label="train_box", marker="o")
if "val_box_loss" in df.columns:
    ax4.plot(df["epoch"], df["val_box_loss"], label="val_box", marker="s")
ax4.set(xlabel="Эпоха", ylabel="box loss", title="Box regression loss")
ax4.legend(); ax4.grid(alpha=.3)

c3, c4 = st.columns(2)
c3.pyplot(fig3)
c4.pyplot(fig4)
# Экспорт
st.markdown("---")
if st.button("📥 Скачать данные (CSV)"):
    csv = df.to_csv(index=False)
    st.download_button("💾 Скачать", csv, "/home/adminadmin/brain_dir/CV_fd_deploy/assets/results.csv", "text/csv")


st.markdown("---")
with st.expander("Предыдущие попытки обучения модели"):
    st.image(str(ROOT / "/home/adminadmin/brain_dir/CV_fd_deploy/assets/training_results.png"), caption="training_results.png")
