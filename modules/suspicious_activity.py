from pathlib import Path
import io
import os
from urllib.parse import urlsplit, urlunsplit
import wave

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
PERSON_CLASS_ID = 0
CAMERA_FRAME_MAX_WIDTH = 960
CAMERA_OPEN_TIMEOUT_MS = 5000
CAMERA_READ_TIMEOUT_MS = 5000
CAMERA_WARMUP_MAX_FRAMES = 8
CAMERA_BLACK_FRAME_THRESHOLD = 5.0
CAMERA_SOLID_FRAME_STD_THRESHOLD = 0.5
LOCAL_CAMERA_BACKENDS = {
    "Any": cv2.CAP_ANY,
    "MSMF": cv2.CAP_MSMF,
    "DirectShow": cv2.CAP_DSHOW,
}


def load_person_detector():
    root_model_path = PROJECT_ROOT / "yolo11n.pt"
    model_path = next((path for path in (LOCAL_MODEL_PATH, root_model_path) if path.exists()), None)
    model_source = str(model_path) if model_path else "yolo11n.pt"
    return YOLO(model_source)


def detect_frame(
    frame_rgb: np.ndarray,
    person_model,
    confidence_threshold=0.18,
    knife_class_id=None,
    knife_confidence_threshold=0.65,
    image_size=640,
    iou_threshold=0.7,
    max_detections=300,
):
    """Detect all model classes; selected_class_ids is retained for API compatibility only."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = person_model.predict(
        source=frame_bgr,
        classes=None,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        max_det=max_detections,
        verbose=False,
    )[0]
    inferred_detections = _result_detections(result, person_model.names)
    all_detections = [
        detection for detection in inferred_detections
        if detection["confidence"] >= confidence_threshold
    ]
    person_count = sum(detection["class_id"] == PERSON_CLASS_ID for detection in all_detections)
    visible_detections = all_detections

    knife_detections = [
        detection for detection in inferred_detections
        if knife_class_id is not None
        and detection["class_id"] == knife_class_id
        and detection["confidence"] >= confidence_threshold
        and detection["confidence"] >= knife_confidence_threshold
    ]
    annotated_detections = visible_detections + [
        detection for detection in knife_detections
        if detection not in visible_detections
    ]
    annotated_rgb = annotate_detections(
        frame_rgb,
        annotated_detections,
        knife_class_id=knife_class_id,
    )
    return annotated_rgb, person_count, visible_detections, knife_detections


def annotate_detections(frame_rgb, detections, knife_class_id=None):
    annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for detection in detections:
        color = _class_color(detection["class_id"])
        if detection["class_id"] == knife_class_id:
            color = (30, 80, 235)
        _draw_detection(annotated_bgr, detection, color)

    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def _class_color(class_id):
    return (
        int((37 * class_id + 83) % 200 + 40),
        int((67 * class_id + 131) % 200 + 40),
        int((97 * class_id + 59) % 200 + 40),
    )


def _draw_detection(image_bgr, detection, color):
    x1, y1, x2, y2 = detection["box"]
    label = f"{detection['class_name']} {detection['confidence']:.1%}"
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _result_detections(result, names):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        coordinates = tuple(int(value) for value in box.xyxy[0].tolist())
        detections.append({
            "class_id": class_id,
            "class_name": str(class_name),
            "confidence": float(box.conf.item()),
            "box": coordinates,
        })
    return detections


def detect_people(image: Image.Image, model):
    image_rgb = np.asarray(image.convert("RGB"))
    _, person_count, detections, _ = detect_frame(image_rgb, model)
    people = [detection for detection in detections if detection["class_id"] == PERSON_CLASS_ID]
    return annotate_detections(image_rgb, people), person_count


def open_camera(source, warmup_frames=CAMERA_WARMUP_MAX_FRAMES, backend="MSMF"):
    capture = None
    try:
        if isinstance(source, str):
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAMERA_OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAMERA_READ_TIMEOUT_MS,
            ]
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        else:
            backend_code = LOCAL_CAMERA_BACKENDS.get(backend)
            if backend_code is None:
                raise ValueError(f"Unsupported local camera backend: {backend}")
            capture = cv2.VideoCapture(int(source), backend_code)
        if not capture.isOpened():
            raise RuntimeError(
                "Camera could not be opened. For a mobile or virtual camera stream, verify the exact URL and protocol shown by the camera app or Iriun Webcam driver, then try the Windows device index that actually exposes a live RGB frame."
            )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_rgb = None
        last_frame_issue = None
        previous_valid_frame = None
        content_changed = False
        frames_read = 0
        for _ in range(max(1, warmup_frames)):
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                if frame_rgb is None:
                    raise RuntimeError(
                        "Camera opened but frame read failed. Confirm that the source camera app or Iriun Webcam virtual camera is actively serving a compatible live video stream."
                    )
                break
            frame_rgb = _prepare_camera_frame(frame)
            frames_read += 1
            last_frame_issue = camera_frame_issue(frame_rgb)
            if last_frame_issue is None:
                if previous_valid_frame is not None and not camera_frames_unchanged(previous_valid_frame, frame_rgb):
                    content_changed = True
                previous_valid_frame = frame_rgb.copy()
                break
        if frame_rgb is None:
            raise RuntimeError("Camera opened but did not provide a valid frame.")
        if last_frame_issue is not None:
            height, width = frame_rgb.shape[:2]
            content_status = "yes" if content_changed else "no usable image content"
            raise RuntimeError(
                f"{last_frame_issue} Startup diagnostics: read {frames_read} frame(s), "
                f"{width}x{height}, content changing: {content_status}."
            )
        return capture, frame_rgb
    except Exception:
        if capture is not None:
            capture.release()
        raise


def probe_local_cameras(device_indices=range(4), frame_count=4):
    device_indices = tuple(int(index) for index in device_indices)
    frame_count = max(1, int(frame_count))
    diagnostics = []
    for backend_name, backend_code in LOCAL_CAMERA_BACKENDS.items():
        for device_index in device_indices:
            capture = None
            frames = []
            read_success = True
            error = None
            opened = False
            try:
                capture = cv2.VideoCapture(int(device_index), backend_code)
                opened = bool(capture.isOpened())
                if not opened:
                    error = "Device did not open with this backend."
                else:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    for _ in range(max(1, frame_count)):
                        success, frame = capture.read()
                        if not success or frame is None or frame.size == 0:
                            read_success = False
                            error = "Frame read failed."
                            break
                        frames.append(_prepare_camera_frame(frame))
            except Exception as exception:
                read_success = False
                error = str(exception)
            finally:
                if capture is not None:
                    capture.release()

            valid_frames = [frame for frame in frames if camera_frame_issue(frame) is None]
            content_changed = any(
                not camera_frames_unchanged(previous, current)
                for previous, current in zip(valid_frames, valid_frames[1:])
            )
            last_frame = frames[-1] if frames else None
            if not opened:
                status = "not opened"
            elif not frames:
                status = "no readable frames"
            elif not valid_frames:
                frame_issue = camera_frame_issue(last_frame)
                status = f"no usable image content: {frame_issue}"
            elif content_changed:
                status = "usable image content; frames changing"
            else:
                status = "image content present; frame changes not verified"

            diagnostics.append({
                "Device index": int(device_index),
                "Backend": backend_name,
                "Opened": opened,
                "Frames read": len(frames),
                "Reads complete": bool(read_success and len(frames) == max(1, frame_count)),
                "Frame size": f"{last_frame.shape[1]}x{last_frame.shape[0]}" if last_frame is not None else "n/a",
                "Mean brightness": round(float(np.mean(last_frame)), 1) if last_frame is not None else None,
                "Valid image frames": len(valid_frames),
                "Content changing": content_changed,
                "Status": status,
                "Error": error or "",
            })
    return diagnostics


def read_camera_frame(capture):
    success, frame = capture.read()
    if not success or frame is None or frame.size == 0:
        raise RuntimeError("Camera stream stopped or returned an invalid frame. Restart the camera connection.")
    frame_rgb = _prepare_camera_frame(frame)
    issue = camera_frame_issue(frame_rgb)
    if issue is not None:
        raise RuntimeError(issue)
    return frame_rgb


def camera_frame_issue(frame_rgb):
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        return "Camera returned an invalid frame shape; expected an RGB image with three channels."
    if frame_rgb.dtype != np.uint8:
        return f"Camera returned {frame_rgb.dtype} pixels; expected uint8 RGB data."
    if frame_rgb.size == 0:
        return "Camera returned an empty frame."
    if float(np.mean(frame_rgb)) < CAMERA_BLACK_FRAME_THRESHOLD:
        return "Camera frame is nearly black. Check the camera app, lens, and lighting."
    channel_std = np.std(frame_rgb.reshape(-1, 3), axis=0)
    if float(np.max(channel_std)) < CAMERA_SOLID_FRAME_STD_THRESHOLD:
        color = tuple(int(value) for value in frame_rgb[0, 0])
        return (
            f"Camera is returning a solid-color frame (RGB {color}) instead of live video. "
            "Check that the camera app is sending its active video feed, then restart the camera."
        )
    return None


def camera_frames_unchanged(previous_frame, current_frame):
    return (
        previous_frame is not None
        and previous_frame.shape == current_frame.shape
        and np.array_equal(previous_frame, current_frame)
    )


def _prepare_camera_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    if width > CAMERA_FRAME_MAX_WIDTH:
        target_height = round(height * CAMERA_FRAME_MAX_WIDTH / width)
        frame_rgb = cv2.resize(frame_rgb, (CAMERA_FRAME_MAX_WIDTH, target_height), interpolation=cv2.INTER_AREA)
    return frame_rgb


def camera_diagnostics(capture, frame_rgb, source, read_success=True):
    try:
        backend = capture.getBackendName()
    except cv2.error:
        backend = "unknown"
    height, width = frame_rgb.shape[:2]
    return {
        "source": redact_camera_source(source),
        "backend": backend,
        "frame_dimensions": f"{width} × {height}",
        "frame_read_success": bool(read_success and frame_rgb.size > 0),
        "mean_brightness": float(np.mean(frame_rgb)),
    }


def redact_camera_source(source):
    if not isinstance(source, str):
        return f"local camera device {int(source)}"
    parts = urlsplit(source)
    if not parts.scheme or not parts.hostname:
        return source
    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def should_play_alarm(enabled, detection_active, current_time, last_alarm_time, cooldown_seconds):
    if not enabled or not detection_active:
        return False
    return last_alarm_time is None or current_time - last_alarm_time >= cooldown_seconds


def claim_alert_event(event_key, recent_event_keys, max_events=128):
    if event_key in recent_event_keys:
        return False
    if isinstance(recent_event_keys, set):
        recent_event_keys.add(event_key)
    else:
        recent_event_keys.append(event_key)
        del recent_event_keys[:-max_events]
    return True


def create_alarm_audio():
    sample_rate = 16000
    duration = 0.45
    sample_count = int(sample_rate * duration)
    times = np.arange(sample_count, dtype=np.float32) / sample_rate
    envelope = np.minimum(times * 20, 1) * np.minimum((duration - times) * 20, 1)
    signal = 0.35 * np.sin(2 * np.pi * 880 * times) * envelope
    samples = np.asarray(signal * 32767, dtype=np.int16)
    audio = io.BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    return audio.getvalue()


def crowding_rule_triggered(person_count: int, threshold: int) -> bool:
    return person_count >= threshold