# Lenta PriceTag Detection

Проект для детекции ценников на изображениях и видео с использованием YOLO.

Модель находит ценники на кадре, выделяет bounding boxes, сохраняет лучшие crop-изображения найденных ценников и может использоваться как отдельный модуль для дальнейшей обработки, например OCR.

## Возможности

- Детекция ценников на изображениях
- Детекция и трекинг ценников на видео
- Сохранение лучших crop-изображений по каждому track ID
- Поддержка YOLO-модели через `ultralytics`
- Поддержка ByteTrack-трекинга
- Фотометрические аугментации для обучения
- Возможность интеграции с OCR-пайплайном

## Стек

- Python 3.12
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib
- Albumentations
- SciPy
- PyYAML

## Структура проекта

```text
Lenta-PriceTag-Detection/
├── configs/
│   ├── crf.yaml              # CRF-конфиг для фотометрической аугментации
│   └── data.yaml             # Конфиг датасета YOLO
├── weights/
│   ├── best.pt               # YOLO-веса PyTorch
│   └── best.onnx             # YOLO-веса в ONNX-формате
├── adapter.py                # Пример интеграции детекции с OCR-пайплайном
├── detection.py              # Основной класс DetectorYOLO
├── photometric_augm.py       # Фотометрическая аугментация изображений
├── price_detection_clean.ipynb
├── requirements.txt
└── README.md
```

## Установка

Склонируйте репозиторий:

```bash
git clone https://github.com/Slipernik/Lenta-PriceTag-Detection.git
cd Lenta-PriceTag-Detection
```

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активируйте окружение.

Для Windows:

```bash
.venv\Scripts\activate
```

Для Linux / macOS:

```bash
source .venv/bin/activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Быстрый старт

В репозитории уже есть обученные веса:

```text
weights/best.pt
weights/best.onnx
```

Основной класс для детекции находится в файле:

```text
detection.py
```

Он называется:

```python
DetectorYOLO
```

## Детекция на изображении

Пример запуска детекции на одном изображении или на папке с изображениями:

```python
from detection import DetectorYOLO

model_path = "weights/best.pt"
tracker = "bytetrack.yaml"
img_path = "path/to/image_or_folder"

detector = DetectorYOLO(
    model_path=model_path,
    tracker=tracker,
    save_root_dir="saved_crops",
    rotate=False,
    frame_skip=1,
    q_crops=5,
    conf=0.1,
    draw_conf=0.5,
    imgsz=960,
    max_det=200,
    device="cuda",  # замените на "cpu", если нет GPU
)

for fname, img, xyxys, confs, clss in detector.image_detection(
    img_path=img_path,
    result_show=True,
):
    print("File:", fname)
    print("Boxes:", xyxys)
    print("Confidences:", confs)
    print("Classes:", clss)
```

Если у вас нет CUDA/GPU, используйте:

```python
device="cpu"
```

## Детекция на видео

Пример обработки видео:

```python
from detection import DetectorYOLO

model_path = "weights/best.pt"
tracker = "bytetrack.yaml"
video_path = "videos/1.mp4"

detector = DetectorYOLO(
    model_path=model_path,
    tracker=tracker,
    save_root_dir="saved_crops",
    rotate=True,
    frame_skip=3,
    q_crops=5,
    conf=0.5,
    draw_conf=0.5,
    imgsz=960,
    max_det=200,
    device="cuda",
)

for frame_idx, frame, grade_d in detector.video_detection(
    video_path=video_path,
    video_save=False,
    video_show=False,
):
    print("Frame:", frame_idx)
    print("Tracks:", grade_d.keys())
```

Во время обработки видео класс сохраняет лучшие crop-изображения ценников в папку, указанную в `save_root_dir`.

Пример:

```text
saved_crops/
└── sequence_1/
    ├── track_1/
    │   ├── 120.jpg
    │   └── 120.txt
    └── track_2/
        ├── 135.jpg
        └── 135.txt
```

## Основные параметры `DetectorYOLO`

| Параметр | Описание |
|---|---|
| `model_path` | Путь к весам модели YOLO |
| `tracker` | Конфиг трекера, например `bytetrack.yaml` |
| `frame_skip` | Обрабатывать каждый N-й кадр |
| `verbose` | Выводить подробные логи |
| `rotate` | Поворачивать кадр на 90° против часовой стрелки |
| `save_root_dir` | Папка для сохранения crop-изображений |
| `q_crops` | Количество лучших crop-изображений для каждого track ID |
| `pad` | Отступ вокруг bounding box при сохранении crop |
| `device` | Устройство: `cuda` или `cpu` |
| `conf` | Минимальный confidence для модели |
| `draw_conf` | Confidence для отрисовки bounding box |
| `iou` | IoU threshold |
| `max_det` | Максимальное количество детекций |
| `imgsz` | Размер изображения для YOLO |

## Обучение модели

В проекте используется формат датасета YOLO.

Конфиг датасета находится здесь:

```text
configs/data.yaml
```

Ожидаемая структура датасета:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Класс объекта:

```text
price-tag
```

Пример обучения:

```python
from ultralytics import YOLO
from photometric_augm import Exposition

custom_transforms = [
    Exposition(
        crf_path="configs/crf.yaml",
        p=0.3,
    )
]

model = YOLO("yolo11s.pt")

model.train(
    data="configs/data.yaml",
    epochs=100,
    imgsz=960,
    device="cuda",
    batch=4,
    nbs=32,
    workers=4,
    optimizer="AdamW",
    seed=1,
    cos_lr=True,
    lr0=1e-3,
    hsv_s=0.35,
    hsv_v=0.25,
    degrees=10,
    translate=0.03,
    perspective=0.0003,
    scale=0.3,
    shear=10,
    fliplr=0.0,
    erasing=0.0,
    mosaic=0.5,
    cutmix=0.5,
    close_mosaic=5,
    save_crop=True,
    nms=True,
    augmentations=custom_transforms,
)
```

После обучения лучшие веса обычно сохраняются в папке `runs/`.

## Фотометрическая аугментация

Файл `photometric_augm.py` содержит класс `Exposition`, который имитирует изменение экспозиции изображения.

Он использует CRF-кривую из файла:

```text
configs/crf.yaml
```

Пример использования:

```python
from photometric_augm import Exposition

augmentation = Exposition(
    crf_path="configs/crf.yaml",
    p=0.3,
)
```

## OCR-интеграция

Файл `adapter.py` показывает пример интеграции детекции с OCR-пайплайном.

Он запускает детекцию ценников на видео, сохраняет лучшие crop-изображения, а затем передаёт их во внешний OCR-скрипт.

Важно: для полной работы OCR-части нужны дополнительные файлы, которых сейчас нет в репозитории:

```text
run_price_tag_pipeline.py
run_detected_tracks_dataset.py
configs/rail_pipeline.yaml
configs/detected_tracks_dataset.yaml
```

Если вам нужна только детекция ценников, `adapter.py` можно не использовать. Достаточно работать с `DetectorYOLO` из файла `detection.py`.

## Работа с Jupyter Notebook

В проекте есть notebook:

```text
price_detection_clean.ipynb
```

В нём собраны примеры:

- импорта модели;
- обучения YOLO;
- запуска детекции на изображениях;
- запуска детекции на видео;
- использования crop-логики для дальнейшего OCR.

Запуск:

```bash
jupyter notebook price_detection_clean.ipynb
```

или:

```bash
jupyter lab
```

## ONNX-веса

В папке `weights/` есть файл:

```text
best.onnx
```

Он может использоваться для инференса вне PyTorch, например в ONNX Runtime, OpenCV DNN или других inference-runtime окружениях.

Текущий код `DetectorYOLO` использует `ultralytics.YOLO`, поэтому основной вариант запуска — через:

```text
weights/best.pt
```

## Результат работы

После обработки изображения или видео можно получить:

- bounding boxes найденных ценников;
- confidence score для каждой детекции;
- class ID;
- crop-изображения ценников;
- набор лучших crop-изображений для каждого track ID.

Пример результата:

```text
saved_crops/
└── sequence_1/
    └── track_5/
        ├── 240.jpg
        ├── 240.txt
        ├── 243.jpg
        └── 243.txt
```

`.jpg` — сохранённый crop ценника.  
`.txt` — значение оценки качества crop-изображения.