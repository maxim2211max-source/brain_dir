import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Конфигурация страницы (только для главной страницы в многостраничном приложении)
st.set_page_config(page_title="Brain MRI Analysis", page_icon="🧠", layout="wide")

# Пути к моделям (абсолютные, как в вашем коде)
MODEL_PATHS = {
    'axial': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp14/weights/best.pt',
    'coronal': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp23/weights/best.pt',
    'sagittal': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp32/weights/best.pt',
    'union': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp_union/weights/best.pt'
}


PLOT_PATHS = {
    'axial': {
        'map': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp14/results.png',
        'pr_curve': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp14/BoxPR_curve.png',
        'confusion_matrix': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp14/confusion_matrix.png'
    },
    'coronal': {
        'map': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp23/results.png',
        'pr_curve': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp23/BoxPR_curve.png',
        'confusion_matrix': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp23/confusion_matrix.png'
    },
    'sagittal': {
        'map': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp32/results.png',
        'pr_curve': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp32/BoxPR_curve.png',
        'confusion_matrix': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp32/confusion_matrix.png'
    },
    'union': {
        'map': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp_union/results.png',
        'pr_curve': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp_union/BoxPR_curve.png',
        'confusion_matrix': '/home/adminadmin/brain_dir/dir_brain_tessssst/train_exp_union/confusion_matrix.png'
    }
}

# Число эпох для каждой модели
EPOCHS = {
    'axial': 31,
    'coronal': 33,
    'sagittal': 41,
    'union': 27
}

# Кэшируем модели
@st.cache_resource
def load_model(plane):
    model_path = MODEL_PATHS[plane]
    return YOLO(model_path)

# Функция для загрузки изображения по URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return np.array(image)
    except Exception as e:
        st.error(f"Ошибка загрузки изображения по URL: {str(e)}")
        return None

st.title("YOLO Модель для анализа МРТ мозга (3 плоскости + объединённая)")

# Селектор плоскости
plane = st.radio(
    "Выберите плоскость МРТ или объединённую модель:",
    options=['axial', 'coronal', 'sagittal', 'union'],
    index=0,
    horizontal=True
)

# Раздел для графиков обучения
if st.checkbox(f"Показать графики обучения для модели '{plane}'"):
    st.subheader(f"Графики обучения модели ({plane})")
    st.write(f"Модель обучена на **{EPOCHS[plane]} эпохах**.")
    
    tab1, tab2, tab3 = st.tabs(["mAP", "PR Кривая", "Confusion Matrix"])
    
    with tab1:
        try:
            map_image = Image.open(PLOT_PATHS[plane]['map'])
            st.image(map_image, caption=f"mAP график ({plane})", use_container_width=True)
        except FileNotFoundError:
            st.error(f"Файл mAP для {plane} не найден. Проверьте путь: {PLOT_PATHS[plane]['map']}")
        except Exception as e:
            st.error(f"Ошибка загрузки mAP: {str(e)}")
    
    with tab2:
        try:
            pr_image = Image.open(PLOT_PATHS[plane]['pr_curve'])
            st.image(pr_image, caption=f"PR Кривая ({plane})", use_container_width=True)
        except FileNotFoundError:
            st.error(f"Файл PR-кривой для {plane} не найден. Проверьте путь: {PLOT_PATHS[plane]['pr_curve']}")
        except Exception as e:
            st.error(f"Ошибка загрузки PR-кривой: {str(e)}")
    
    with tab3:
        try:
            cm_image = Image.open(PLOT_PATHS[plane]['confusion_matrix'])
            st.image(cm_image, caption=f"Confusion Matrix ({plane})", use_container_width=True)
        except FileNotFoundError:
            st.error(f"Файл Confusion Matrix для {plane} не найден. Проверьте путь: {PLOT_PATHS[plane]['confusion_matrix']}")
        except Exception as e:
            st.error(f"Ошибка загрузки Confusion Matrix: {str(e)}")

# Загрузка модели
try:
    model = load_model(plane)
except Exception as e:
    st.error(f"Ошибка загрузки модели для {plane}: {str(e)}")
    st.stop()

# Загрузка изображений
st.subheader("Загрузка изображений")
uploaded_files = st.file_uploader("Загрузите изображения МРТ (JPG/PNG) — поддерживается множественная загрузка", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
url_input = st.text_input("Или введите прямую ссылку на изображение (для одного файла)")

images = []
image_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        images.append(image)
        image_names.append(uploaded_file.name)

if url_input:
    image_from_url = load_image_from_url(url_input)
    if image_from_url is not None:
        if len(image_from_url.shape) == 2:
            image_from_url = cv2.cvtColor(image_from_url, cv2.COLOR_GRAY2RGB)
        images.append(cv2.cvtColor(image_from_url, cv2.COLOR_RGB2BGR))
        image_names.append("Изображение по URL")

if not images:
    st.info("Загрузите файлы или введите URL для начала анализа.")
else:
    st.subheader("Оригинальные изображения")
    cols = st.columns(min(len(images), 3))
    for i, (img, name) in enumerate(zip(images, image_names)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols[i % 3].image(img_rgb, caption=name, use_container_width=True)

    conf_threshold = st.slider("Порог уверенности (conf)", 0.1, 0.9, 0.25, 0.05)

    if st.button("Запустить предсказание для всех изображений"):
        for idx, (image, name) in enumerate(zip(images, image_names)):
            st.subheader(f"Результаты для {name} ({plane})")
            with st.spinner(f"Предсказание для {name}..."):
                results = model.predict(image, conf=conf_threshold, device=0, verbose=False)

            result = results[0]

            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                st.success(f"Обнаружено {len(boxes)} объектов!")
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf_score = float(boxes.conf[i])
                    class_name = model.names[cls]
                    st.write(f"Объект {i+1}: Класс '{class_name}' (ID: {cls}), Уверенность: {conf_score:.2f}")
                
                annotated = result.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="С аннотациями (детекция)", use_container_width=True)
            else:
                st.warning("❌ Нет детекций! Попробуйте понизить conf или проверьте изображение/модель.")
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Без аннотаций (пустой результат)", use_container_width=True)

            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                st.success(f"Обнаружено {len(result.masks)} масок сегментации!")
                annotated_seg = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_seg, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="С масками (сегментация)", use_container_width=True)
            else:
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    st.info("Если это сегментация, masks тоже пустые. Проверьте тип задачи в YAML.")

            if st.checkbox(f"Показать сырой объект results для {name} (для отладки)"):
                st.code(str(result.boxes), language='python')
