from pathlib import Path
import json
import subprocess
import sys
import time
from multiprocessing import Process, Queue

from detection import DetectorYOLO


PROJECT_DIR = Path(__file__).resolve().parent

video_path = PROJECT_DIR / "videos/1.mp4"
model_path = PROJECT_DIR / "weights/best.pt"
tracker = "bytetrack.yaml"

ROOT_DIR = PROJECT_DIR

work_dir = PROJECT_DIR / "runs/basic_joint"
crops_dir = work_dir / "crops"
ocr_dir = work_dir / "ocr"

frame_diff = 150

SCRIPT_PATH = ROOT_DIR / "run_price_tag_pipeline.py"
CONFIG_PATH = ROOT_DIR / "configs/rail_pipeline.yaml"

Q_CROPS = 5

def run_lenta_single_tag_ocr(crop_path: Path, out_dir: Path):
    crop_path = Path(crop_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--config",
        str(CONFIG_PATH),
        "--image",
        str(crop_path),
        "--input_mode",
        "single_tag",
        "--out_dir",
        str(out_dir),
    ]

    print("[OCR CMD]", " ".join(cmd), flush=True)

    t0 = time.time()

    subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        check=True,
    )

    print(f"[OCR TIME] {crop_path.name}: {time.time() - t0:.2f} sec", flush=True)

    json_files = list(out_dir.rglob("*summary*.json")) + list(out_dir.rglob("*.json"))

    if not json_files:
        print(f"[OCR DONE] JSON result not found in {out_dir}", flush=True)
        return None

    json_path = sorted(json_files, key=lambda p: len(p.parts))[0]

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return result

def run_lenta_sequence_tag_ocr(root_dir: Path, out_dir: Path):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        'run_detected_tracks_dataset.py',
        "--config",
        'configs/detected_tracks_dataset.yaml',
        "--root_dir",
        str(root_dir),
        "--out_dir",
        str(out_dir),
    ]

    print("[OCR CMD]", " ".join(cmd), flush=True)

    t0 = time.time()

    subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        check=True,
    )

    print(f"[OCR TIME] sequence: {time.time() - t0:.2f} sec", flush=True)

    json_files = (
        list(out_dir.rglob("*results*.json"))
        + list(out_dir.rglob("*summary*.json"))
        + list(out_dir.rglob("*.json"))
    )

    if not json_files:
        print(f"[OCR DONE] JSON result not found in {out_dir}", flush=True)
        return None

    json_path = sorted(json_files, key=lambda p: (len(p.parts), p.name))[0]

    print(f"[OCR JSON] {json_path}", flush=True)

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)



det = DetectorYOLO(
    model_path=str(model_path),
    tracker=tracker,
    save_root_dir='results',
    rotate=True,
    frame_skip=1,
    q_crops=Q_CROPS,
    conf=0.5,
    imgsz=960,
    max_det=200,
)

results = {}


for frame_idx, img, grade_d in det.video_detection(
    video_path=video_path,
    video_save=False,
    video_show=False
):
    if frame_idx % det.frame_skip == 0:        
        for tr_id, vals in list(grade_d.items()):
            if len(vals) > 0:
                frame_last = max(vals, key= lambda x: x['frame_idx'])['frame_idx']

            if frame_idx - frame_last >= frame_diff:
     
                best = max(vals, key=lambda x: x['grade'])
                crop_path = best['path']
                
                run_lenta_single_tag_ocr(crop_path, Path(crop_path).parent)                

                # run_lenta_sequence_tag_ocr(Path('results'), Path(crop_path).parent.parent)

                det.clear_track(tr_id)

