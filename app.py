import json
import html
import base64
import time
import hashlib
import tempfile
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer
import av
from pathlib import Path
from detection import DetectorYOLO


@st.cache_resource(show_spinner=False)
def model_init(
    frame_skip=1,
    conf=0.5,
    rotate=False,
    q_crops=5,
    imgsz=960,
    device='cpu',    
    alpha=0.5,
    cache_buster=0,
    ):

    model = DetectorYOLO(
        model_path='weights/best.pt',
        tracker='bytetrack.yaml',
        frame_skip=frame_skip,
        rotate=rotate,
        q_crops=q_crops,
        device=device,
        conf=conf,
        imgsz=imgsz,
        max_det=200,
        alpha=alpha,
        save_tracks_csv=True,
        tracks_csv_path="tracks.csv",        
    )

    return model

def bgr_to_html_color(color_bgr):
    b, g, r = color_bgr

    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    return f"rgb({r}, {g}, {b})"

def _crop_to_base64(crop_bgr, max_side=180):
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    crop = crop_bgr.copy()

    h, w = crop.shape[:2]
    scale = min(max_side / max(h, w), 1.0)

    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )

    if not ok:
        return None

    return base64.b64encode(buffer).decode("utf-8")

def resize_video_frame(img):
    H, W = img.shape[:2]

    if W < H:
        resize = (432, 768)
    else:
        resize = (1152, 768)

    return cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

def get_video_layout(file_path, rotate):
    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        return [1, 2]

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return [1, 2]

        H, W = frame.shape[:2]
        if rotate:
            H, W = W, H

        return [1, 2] if H > W else [2, 1]
    finally:
        cap.release()


def on_rotate_change():
    st.session_state.rotate_video = st.session_state.rotate_video_widget
    st.session_state.video_play = False
    st.session_state.track_badges = {}
    st.session_state.model_version += 1

def _format_json_like_text(text):
    if text is None:
        return ""

    if isinstance(text, str):
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
    else:
        text = json.dumps(text, ensure_ascii=False, indent=2)

    return html.escape(str(text))

def update_track_badge_panel(
    placeholder,
    crop_bgr,
    frame_idx,
    track_id,
    text=None,
    state_key="track_badges",
    ttl_frames=None,
    max_badges=100,
    panel_height=760,
    img_mode=False
):
    if state_key not in st.session_state:
        st.session_state[state_key] = {}

    badges = st.session_state[state_key]

    b64 = _crop_to_base64(crop_bgr)

    if b64 is not None:
        badges[str(track_id)] = {
            "track_id": track_id,
            "frame_idx": int(frame_idx),
            "image": b64,
            "text": _format_json_like_text(text),
        }

    if ttl_frames is not None:
        stale_ids = [
            tr_id
            for tr_id, item in badges.items()
            if int(frame_idx) - item["frame_idx"] > ttl_frames
        ]

        for tr_id in stale_ids:
            badges.pop(tr_id, None)

    sorted_badges = sorted(
        badges.values(),
        key=lambda x: x["track_id"],
        reverse=False,
    )[:max_badges]

    cards = []

    for item in sorted_badges:
        tr_id = item["track_id"]
        if img_mode is False:
            color = bgr_to_html_color(model.get_color(tr_id))
        else:
            color = bgr_to_html_color(model.get_color(0))

        cards.append(f"""
        <div class="track-card" style="border-color: {color};">
            <div class="track-image-wrap">
                <img src="data:image/jpeg;base64,{item['image']}" />
            </div>

            <div class="track-info">
                <div class="track-title" style="color: {color};">
                    track_id: {html.escape(str(tr_id))}
                </div>
                <div class="track-frame">
                    frame: {item["frame_idx"]}
                </div>
                <pre class="track-text">{item["text"]}</pre>
            </div>
        </div>
        """)

    html_block = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: sans-serif;
            }}

            .track-panel {{
                width: 100%;
                height: {panel_height}px;
                overflow-y: auto;
                padding: 6px;
                box-sizing: border-box;

                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                gap: 10px;
                align-items: start;
            }}

            .track-card {{
                display: flex;
                gap: 8px;
                border: 3px solid;
                border-radius: 12px;
                padding: 8px;
                background: #111;
                box-sizing: border-box;
                min-width: 0;
            }}

            .track-image-wrap {{
                width: 90px;
                min-width: 90px;
                height: 90px;
                background: #222;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }}

            .track-image-wrap img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                display: block;
            }}

            .track-info {{
                flex: 1;
                min-width: 0;
            }}

            .track-title {{
                font-size: 14px;
                font-weight: 700;
                margin-bottom: 2px;
            }}

            .track-frame {{
                color: #aaa;
                font-size: 12px;
                margin-bottom: 6px;
            }}

            .track-text {{
                margin: 0;
                color: #eee;
                font-size: 12px;
                line-height: 1.25;
                white-space: pre-wrap;
                word-break: break-word;
                overflow-wrap: anywhere;
                max-height: 78px;
                overflow-y: auto;
                background: #1b1b1b;
                border-radius: 6px;
                padding: 6px;
                box-sizing: border-box;
            }}
        </style>
    </head>

    <body>
        <div class="track-panel">
            {''.join(cards)}
        </div>
    </body>
    </html>
    """

    with placeholder.container():
        components.html(
            html_block,
            height=panel_height + 20,
            scrolling=False,
        )


if "video_play" not in st.session_state:
    st.session_state.video_play = False
    st.session_state.video_wait = False
if "track_badges" not in st.session_state:
    st.session_state.track_badges = {}
if "rotate_video" not in st.session_state:
    st.session_state.rotate_video = "Да"
if "rotate_video_widget" not in st.session_state:
    st.session_state.rotate_video_widget = st.session_state.rotate_video
if "model_version" not in st.session_state:
    st.session_state.model_version = 0
if "current_video_key" not in st.session_state:
    st.session_state.current_video_key = None

rotate = False
frame_diff = 60
q_crops = 5


st.set_page_config(
    page_title='Детекция ценников Лента',
    layout='wide'
)

# ------------------------------------------------------------
# ------------------ Настройка конфигурации ------------------
# ------------------------------------------------------------

with st.sidebar:
    st.header('Настройка конфигурации')
    source = st.radio('Выберите источник', ['Изображение', 'Видео'], index=1)

    if source == "Изображение":
        file_number = st.pills('Примеры', [1, 2, 3, 4, 5], default=1)
        image_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if image_file:
            suffix = Path(image_file.name).suffix

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(image_file.getbuffer())
            tmp.close()

            file_path = tmp.name
        else:
            file_path = f'test_img/{file_number}.png'

        st.divider()

        # ---------- Настройки детекционной модели ----------
        st.header('Настройка детекционной модели')
        frame_skip=1


    elif source == "Видео":
        file_number = st.pills('Примеры', [1, 2, 3, 4, 5], default=1)
        video_file = st.file_uploader("Загрузите видео", type=["mp4", "mov", "avi", "mkv"])
        if video_file:
            file_key = f"{video_file.name}_{video_file.size}"

            if st.session_state.get("uploaded_video_key") != file_key:
                suffix = Path(video_file.name).suffix

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(video_file.getbuffer())
                tmp.close()

                st.session_state.uploaded_video_key = file_key
                st.session_state.uploaded_video_path = tmp.name
                st.session_state.video_play = False
                st.session_state.track_badges = {}

            file_path = st.session_state.uploaded_video_path
            current_video_key = file_key
        else:
            file_path = f"test_video/{file_number}.mp4"
            current_video_key = f"sample_{file_number}"

        if st.session_state.current_video_key != current_video_key:
            st.session_state.current_video_key = current_video_key
            st.session_state.video_play = False
            st.session_state.track_badges = {}
            st.session_state.model_version += 1

        if st.session_state.video_play and st.session_state.video_wait == False:
            if st.button("◼", key="pause_btn"):
                st.session_state.video_play = False
                st.rerun() 
        else:
            if st.button("▶", key="play_btn"):
                st.session_state.video_play = True
                st.rerun()
        
        st.divider()

        # ---------- Настройки детекционной модели ----------
        st.header('Настройка детекционной модели')

        # ------- Поворот -------
        if st.session_state.rotate_video_widget != st.session_state.rotate_video:
            st.session_state.rotate_video_widget = st.session_state.rotate_video

        rotate = st.segmented_control(
            'Перевернутое видео',
            ['Да', 'Нет'],
            key='rotate_video_widget',
            on_change=on_rotate_change,
        )
        rotate = st.session_state.rotate_video == 'Да'

        # ------- Запуск OCR на каждом N кадре -------
        frame_diff = st.slider(f'Запуск OCR на каждом N кадре', 1, 180, frame_diff)

        # ------- Пропуск кадров -------
        frame_skip = st.slider('Пропуск кадров', 1, 30, 3)

        # ------- Количество лучших crop -------
        q_crops = st.slider('Количество лучших crop', 1, 20, q_crops)
   
    # ------- Вероятность -------
    probs_thresh = st.slider(f'Порог уверенности', 0.0, 1.0, 0.65, step=0.05)

    # ------- Альфа -------
    alpha = 1.0 - st.slider(f'Затемнение фона', 0.0, 1.0, 0.7, step=0.1)

    # ------- Размер входных данных -------
    imgsz = st.slider('Размер входных данных', 240, 960, 960)

    # ------- Девайс -------
    device = st.segmented_control(
        'Девайс',
        ['Авто', 'CPU', 'GPU'],
        default='Авто'
    )
    if device == 'Авто':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'CPU':
        device = 'cpu'
    elif device == 'GPU':
        device = 'cuda'

    model = model_init(
        frame_skip=frame_skip,
        conf=probs_thresh,
        rotate=rotate,
        q_crops=q_crops,
        imgsz=imgsz, 
        device=device,
        alpha=alpha,
        cache_buster=st.session_state.model_version,
    )

if source == "Изображение":
    det_col, ocr_col = st.columns([2, 1])
elif source == "Видео":
    det_col, ocr_col = st.columns(get_video_layout(file_path, rotate))

frame_draw = det_col.empty()
badge_panel = ocr_col.empty()

# ------------------------------------------------------------
# ------------------- Отображение детекции -------------------
# ------------------------------------------------------------

with det_col:    
    if source == "Изображение":
        _, img, result_img, xyxys, confs, _ = model.image_detection(file_path, result_show=True)
        H, W = result_img.shape[:2]
        resize_img = cv2.resize(
            result_img,
            (768, 1152) if H > W else (1152, 768),
        )
        frame_draw.image(
            resize_img,
            channels='BGR',
        )

        for i, (xyxy, conf) in enumerate(zip(xyxys, confs)):
            crop = model.get_crop_np(img, xyxy)
            update_track_badge_panel(
                placeholder=badge_panel,
                crop_bgr=crop,
                frame_idx=0,
                track_id=i,
                text=None,
                ttl_frames=150,
                max_badges=80,
            )

    elif source == "Видео":
        model.grade_d = {}
        if st.session_state.video_play:
            for frame_idx, img, grade_d in model.video_detection(
                video_path=file_path,
                video_save=False,
                video_show=True
            ):              
                img = resize_video_frame(img)
                frame_draw.image(
                    img,
                    channels='BGR',
                )
                if frame_idx % model.frame_skip == 0:
                    for tr_id, vals in list(grade_d.items()):
                        if len(vals) < model.q_crops:
                            continue

                        best = vals[0]
                        frame_last = max(item["last_seen_frame"] for item in vals)
                        crop = best["crop"]

                        if crop is None:
                            continue

                        text_json_like = {
                            "track_id": tr_id,
                            "frame_idx": frame_last,
                            "status": "detected",
                            "ocr": "OCR",
                            "conf": round(float(best.get("grade", 0.0)), 4),
                        }

                        if len(vals) >= model.q_crops and frame_idx - frame_last >= frame_diff:

                            update_track_badge_panel(
                                placeholder=badge_panel,
                                crop_bgr=crop,
                                frame_idx=frame_idx,
                                track_id=tr_id,
                                text=text_json_like,
                                ttl_frames=150,
                                max_badges=80,
                            )
                            model.clear_track(tr_id)

            st.download_button(
                label="Загрузить CSV",
                data=pd.read_csv('tracks.csv').to_csv().encode('utf-8'),
                file_name="tracks.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            st.session_state.video_play = False
        else:
            img = next(model.video_detection(
                video_path=file_path,
                video_save=False,
                video_show=False
            ))[1]
            img = resize_video_frame(img)
            frame_draw.image(
                img,
                channels='BGR',
            )
