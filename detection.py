import csv
import time
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

class DetectorYOLO:
    def __init__(
        self,
        model_path,
        tracker,
        frame_skip=1,
        verbose=False,
        rotate=False,
        save_root_dir=None,
        q_crops=5,
        pad=10,
        device="cuda",
        conf=0.1,
        draw_conf=0.5,
        iou=0.5,
        max_det=200,
        imgsz=960,
        save_tracks_csv=False,
        tracks_csv_path="tracks.csv",
        alpha=0.7,
    ):
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.frame_skip = frame_skip
        self.verbose = verbose
        self.rotate = rotate
        self.save_root_dir = save_root_dir
        self.q_crops = q_crops
        self.pad = pad
        self.device = device
        self.conf = conf
        self.draw_conf = draw_conf
        self.iou = iou
        self.max_det = max_det
        self.imgsz = imgsz
        self.alpha = alpha

        self.save_tracks_csv = save_tracks_csv
        self.tracks_csv_path = Path(tracks_csv_path) if tracks_csv_path else None

        if self.save_tracks_csv and self.tracks_csv_path is not None:
            self.tracks_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tracks_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_idx", "tr_id", "x1", "y1", "x2", "y2"])

        self.grade_d = {}
        self.save_dir = self._init_save_dir(save_root_dir)

        self.model.to(device)
        self.model.eval()

    def tracking(self, frame, frame_idx):
        with torch.inference_mode():
            pred = self.model.track(
                frame,
                tracker=self.tracker,
                persist=True,
                save=False,
                verbose=self.verbose,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
            )

        result = pred[0]
        img = frame.copy()
        img2 = frame.copy()
        boxes = result.boxes
        names = result.names

        H, W = img.shape[:2]

        if self.alpha < 1.0:
            img = img / 255 * self.alpha * 255
            img = img.astype(np.uint8)

        if boxes is None or boxes.id is None or len(boxes) == 0:
            return img

        xyxys = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        clss = boxes.cls.detach().cpu().numpy().astype(np.int32)
        ids = boxes.id.detach().cpu().numpy().astype(np.int32)

        for xyxy, conf, bcls, tr_id in zip(xyxys, confs, clss, ids):
            
            self.save_track_to_csv(frame_idx, tr_id, xyxy)
            crop = self.get_crop_np(img2, xyxy)
            grade = self.calc_grade(crop, conf, xyxy, H, W)

            self.update_best_crops(
                crop=crop,
                xyxy=xyxy,
                grade=grade,
                frame_idx=frame_idx,
                tr_id=tr_id,
            )

            if conf >= self.draw_conf:
                x1, y1, x2, y2 = xyxy.astype(np.int32)
    
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, W)
                y2 = min(y2, H)
                img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
                
                self.draw_bboxes_cv(img, xyxy, grade, bcls, names, tr_id)                

        return img
    
    def predict(self, img, result_show):
        with torch.inference_mode():
            pred = self.model.predict(
                img,
                verbose=self.verbose,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
            )

        result = pred[0]
        img = result.orig_img.copy()
        img2 = result.orig_img.copy()
        boxes = result.boxes
        names = result.names

        H, W = img.shape[:2]

        if self.alpha < 1.0:
            img = img / 255 * self.alpha * 255
            img = img.astype(np.uint8)

        if boxes is None or len(boxes) == 0:
            return img, None, None, None        

        xyxys = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        clss = boxes.cls.detach().cpu().numpy().astype(np.int32)

        for i, (xyxy, conf, bcls) in enumerate(zip(xyxys, confs, clss)):
            if self.save_dir:
                crop = self.get_crop_np(img, xyxy)
                self.save_crop(
                    save_dir=self.save_dir,
                    crop=crop,
                    frame_idx=time.perf_counter(),
                    tr_id=bcls,
                    grade=conf,
                )
            if conf >= self.draw_conf and result_show:
                x1, y1, x2, y2 = xyxy.astype(np.int32)
    
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, W)
                y2 = min(y2, H)
                img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

                self.draw_bboxes_cv(img, xyxy, conf, bcls, names, i)     

        return img, xyxys, confs, clss

    def video_detection(self, video_path, video_save=False, video_show=False):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0:
            fps = 25

        out_size = (height, width) if self.rotate == True else (width, height)

        out = None
        if video_save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                f"{Path(video_path).stem}.mp4",
                fourcc,
                fps,
                out_size,
            )

            if not out.isOpened():
                raise RuntimeError("VideoWriter не открылся")

        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret or frame is None:
                    break

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                unchanged_img = frame.copy()
                #img = unchanged_img


                if frame_idx % self.frame_skip == 0:
                    if self.verbose:
                        print("frame_idx:", frame_idx)

                    img = self.tracking(frame, frame_idx)

                    if self.verbose:
                        print(
                            "grade_d sizes:",
                            {k: len(v) for k, v in self.grade_d.items()},
                        )
                if video_show:
                    yield frame_idx, img, self.grade_d
                else:
                    yield frame_idx, unchanged_img, self.grade_d

                if video_show:
                    cv2.imshow("Live Detection", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if video_save and out is not None:
                    video_save_path = Path('saved_video').mkdir(parents=True, exist_ok=True)
                    out.write(video_save_path / img)

                frame_idx += 1

        finally:
            cap.release()

            if out is not None:
                out.release()

            if video_show:
                cv2.destroyAllWindows()

    def image_detection(self, img_path, result_show=False):
        files = [] 
        support_format = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        img_path = Path(img_path)

        if img_path.is_dir():
            for file in img_path.iterdir():
                if file.suffix.lower() in support_format:
                    files.append(file)
        else:
            files.append(img_path)

        for f in files:
            img = cv2.imread(str(f))

            result_img, xyxys, confs, clss = self.predict(img, result_show)

            if result_show:
                cv2.imshow("Image Detection", result_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
                return f.name, img, result_img, xyxys, confs, clss
            
            else:
                return f.name, img, None, xyxys, confs, clss
            """
            if f == files[-1]:
                return f.name, img, xyxys, confs, clss
            else:
                yield f.name, img, xyxys, confs, clss
            """


        
    # --------------------------------------
    # ------- Отрисовка и сохранение -------
    # --------------------------------------


    def get_color(self, bcls):
        cmap = plt.get_cmap("Paired")
        float_map = cmap(bcls % 12)
        return (
            float_map[2] * 255.0,
            float_map[1] * 255.0,
            float_map[0] * 255.0
        )

    def draw_bboxes_cv(self, img, xyxy, conf, bcls, names, tr_id=None):
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = max(img.shape[0], img.shape[1]) / 2000
        thickness = int(4 * fontScale)

        x1, y1, x2, y2 = self._xyxy_to_np(xyxy).astype(int)

        bcls = self._value_to_int(bcls)
        tr_id = self._value_to_int(tr_id) if tr_id is not None else None

        color = self.get_color(tr_id if tr_id is not None else bcls)

        cv2.rectangle(
            img=img,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color,
            thickness=thickness
        )
        text = names[int(bcls)]

        if tr_id:
            text += f' / {int(tr_id)}'
        
        text += f' / {conf:.2f}'

        (w, h), _ = cv2.getTextSize(
            text,
            fontFace,
            fontScale,
            thickness
        )
        cv2.rectangle(
            img,
            (x1, y1 - h * 2),
            (x1 + w, y1),
            tuple(int(c * 0.3) for c in color),
            -1
        )

        cv2.putText(
            img=img,
            text=text,
            org=(x1, y1-10),
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness
        )

    def get_crop(self, img, xyxy):
        H, W = img.shape[:2]

        x1, y1, x2, y2 = self._xyxy_to_np(xyxy).astype(int)

        crop = img[
            max(y1 - self.pad, 0):min(y2 + self.pad, H),
            max(x1 - self.pad, 0):min(x2 + self.pad, W)
        ]

        return crop
    
    def get_crop_np(self, img, xyxy):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = xyxy.astype(np.int32)

        x1 = max(x1 - self.pad, 0)
        y1 = max(y1 - self.pad, 0)
        x2 = min(x2 + self.pad, W)
        y2 = min(y2 + self.pad, H)

        if x2 <= x1 or y2 <= y1:
            return None

        return img[y1:y2, x1:x2]
    
    def save_crop(self, save_dir, crop, frame_idx, tr_id, grade=None):
        if save_dir is None:
            return None

        if crop is None or crop.size == 0:
            return None

        if crop.shape[0] * crop.shape[1] <= 6400:
            return None

        id_dir = save_dir / f"id_{tr_id}"
        id_dir.mkdir(parents=True, exist_ok=True)

        img_path = id_dir / f"{frame_idx}.jpg"
        txt_path = id_dir / f"{frame_idx}.txt"

        cv2.imwrite(str(img_path), crop)

        if grade is not None:
            with open(txt_path, "w") as f:
                f.write(str(float(grade)))

        return img_path
    
    def save_track_to_csv(self, frame_idx, tr_id, xyxy):
        if not self.save_tracks_csv or self.tracks_csv_path is None:
            return

        x1, y1, x2, y2 = xyxy # self._xyxy_to_np(xyxy).astype(float)

        with open(self.tracks_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(frame_idx),
                int(tr_id),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ])

    def update_best_crops(self, crop, xyxy, grade, frame_idx, tr_id):
        
        tr_id = int(tr_id)
        grade = float(grade)
        frame_idx = int(frame_idx)

        vals = self.grade_d.setdefault(tr_id, [])

        for item in vals:
            item["last_seen_frame"] = frame_idx

        # если уже есть q_crops и новый хуже или равен худшему — не сохраняем
        if len(vals) >= self.q_crops:
            worst = min(vals, key=lambda x: x["grade"])
            if grade <= worst["grade"]:
                self.grade_d[tr_id] = vals
                return

        img_path = None

        crop_to_store = None if crop is None else crop.copy()

        if self.save_dir is not None:
            img_path = self.save_crop(
                save_dir=self.save_dir,
                crop=crop,
                frame_idx=frame_idx,
                tr_id=tr_id,
                grade=grade,
            )

            if img_path is None:
                return
            
        vals.append(
            {
                "frame_idx": frame_idx,
                "grade": grade,
                "xyxy": np.array(xyxy, dtype=np.float32),
                "crop": crop_to_store,
                "path": None if img_path is None else Path(img_path),
                "last_seen_frame": frame_idx
            }
        )

        # оставляем только q_crops лучших
        vals.sort(key=lambda x: x["grade"], reverse=True)

        while len(vals) > self.q_crops:
            removed = vals.pop(-1)
            self._remove_crop_files(removed["path"])

        self.grade_d[tr_id] = vals
        
    def clear_track(self, tr_id, remove_files=False):
        tr_id = self._value_to_int(tr_id)

        vals = self.grade_d.pop(tr_id, [])

        if remove_files:
            for item in vals:
                self._remove_crop_files(item["path"])

    def _remove_crop_files(self, img_path):
        if img_path is None:
            return

        img_path = Path(img_path)
        txt_path = img_path.with_suffix(".txt")

        if img_path.exists():
            img_path.unlink()

        if txt_path.exists():
            txt_path.unlink()

    def _init_save_dir(self, save_root_dir):
        if not save_root_dir:
            return None

        crops_dir = Path(save_root_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

        try_n = len([d for d in crops_dir.iterdir() if d.is_dir()]) + 1
        save_dir = crops_dir / f"try_{try_n}"
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir

    
    # --------------------------------------
    # --- Расчет оценки для отбора crop ----
    # --------------------------------------


    def grade_sharpness(self, crop):
        if crop is None or crop.size == 0:
            return 0.0
        
        crop = cv2.resize(crop, (64, 64))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        value = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(value / 2000.0, 1.0)

    def grade_area(self, xyxy, H, W):
        if xyxy is None or H <= 0 or W <= 0:
            return 0.0

        x1, y1, x2, y2 = np.array(xyxy, dtype=np.float32)
        bbox_area = max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)
        frame_area = float(H * W)

        if frame_area <= 0:
            return 0.0

        area_ratio = bbox_area / frame_area

        return min(area_ratio / 0.05, 1.0)
    
    def grade_glare(self, crop):
        if crop is None or crop.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ratio = (gray > 215).mean() / 0.2
        return min(ratio, 1.0)
    
    def calc_grade(self, crop, conf, xyxy, H, W):
        if crop is None or crop.size == 0:
            return 0.0
        
        conf = self._value_to_float(conf)

        return float(
            0.35 * self.grade_sharpness(crop)
            + 0.25 * self.grade_area(xyxy, H, W)
            + 0.4 * conf
        )
    

    # ---------------------------------------
    # ------- Преобразование значений -------
    # ---------------------------------------


    def _xyxy_to_np(self, xyxy):
        if hasattr(xyxy, "detach"):
            return xyxy.detach().cpu().numpy().astype(float)
        return np.array(xyxy, dtype=float)

    def _value_to_float(self, x):
        if hasattr(x, "detach"):
            return float(x.detach().cpu().item())
        return float(x)

    def _value_to_int(self, x):
        if hasattr(x, "detach"):
            return int(x.detach().cpu().item())
        return int(x)
    
