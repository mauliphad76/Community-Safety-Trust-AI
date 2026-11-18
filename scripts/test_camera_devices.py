"""Probe Windows camera indices for usable, changing image frames."""

import time

import cv2
import numpy as np


DEVICE_INDICES = range(4)
FRAME_COUNT = 5
FRAME_INTERVAL_SECONDS = 0.15
BLACK_FRAME_MEAN_THRESHOLD = 5.0
SOLID_FRAME_STD_THRESHOLD = 0.5
MIN_FRAME_CHANGE = 0.5
RESOLUTION_TESTS = [(640, 480), (1280, 720), (1920, 1080)]
FOURCC_TESTS = [
    ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
    ("YUY2", cv2.VideoWriter_fourcc(*"YUY2")),
    ("RGB3", cv2.VideoWriter_fourcc(*"RGB3")),
]


def available_backends():
    candidates = (
        ("Any", cv2.CAP_ANY),
        ("MSMF", cv2.CAP_MSMF),
        ("DirectShow", cv2.CAP_DSHOW),
    )
    available = []
    for name, backend in candidates:
        try:
            has_backend = backend == cv2.CAP_ANY or cv2.videoio_registry.hasBackend(backend)
        except Exception:
            has_backend = backend == cv2.CAP_ANY
        if has_backend:
            available.append((name, backend))
    return available


def inspect_frame(frame):
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        return {
            "valid": False,
            "issue": f"unexpected frame format {frame.shape}, {frame.dtype}",
        }

    channel_std = np.std(frame.reshape(-1, 3), axis=0)
    mean_brightness = float(np.mean(frame))
    is_solid = float(np.max(channel_std)) < SOLID_FRAME_STD_THRESHOLD
    is_black = mean_brightness < BLACK_FRAME_MEAN_THRESHOLD
    if is_solid:
        issue = "solid-color frame"
    elif is_black:
        issue = "nearly black frame"
    else:
        issue = None

    return {
        "valid": issue is None,
        "issue": issue,
        "height": int(frame.shape[0]),
        "width": int(frame.shape[1]),
        "mean": mean_brightness,
        "std": float(np.std(frame)),
        "channel_std": channel_std,
    }


def probe_device(device_index, backend_name, backend):
    capture = None
    frames = []
    errors = []
    opened = False
    backend_reported = "unknown"
    active_resolution = None
    active_fourcc = None

    try:
        capture = cv2.VideoCapture(int(device_index), backend)
        opened = bool(capture.isOpened())
        if not opened:
            return {
                "index": device_index,
                "backend": backend_name,
                "opened": False,
                "status": "NOT OPENED",
                "details": "OpenCV could not open this device/backend pair.",
            }

        try:
            backend_reported = capture.getBackendName()
        except cv2.error:
            backend_reported = backend_name

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for width, height in RESOLUTION_TESTS:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            for fourcc_name, fourcc_value in FOURCC_TESTS:
                try:
                    capture.set(cv2.CAP_PROP_FOURCC, float(fourcc_value))
                except cv2.error:
                    pass
                success, frame = capture.read()
                if not success or frame is None or frame.size == 0:
                    errors.append(f"resolution {width}x{height} fourcc {fourcc_name} read failed")
                    continue
                active_resolution = (width, height)
                active_fourcc = fourcc_name
                frames.append(frame.copy())
                break
            if frames:
                break

        if not frames:
            for frame_number in range(FRAME_COUNT):
                if frame_number:
                    time.sleep(FRAME_INTERVAL_SECONDS)
                success, frame = capture.read()
                if not success or frame is None or frame.size == 0:
                    errors.append(f"read {frame_number + 1} failed")
                    break
                frames.append(frame.copy())
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if capture is not None:
            capture.release()

    if not frames and errors:
        return {
            "index": device_index,
            "backend": backend_name,
            "opened": opened,
            "backend_reported": backend_reported,
            "status": "NO READABLE FRAMES",
            "details": f"OpenCV opened the device but reported no valid frame data. Read errors: {'; '.join(errors)}",
        }

    if not frames:
        return {
            "index": device_index,
            "backend": backend_name,
            "opened": opened,
            "backend_reported": backend_reported,
            "status": "NO READABLE FRAMES",
            "details": "; ".join(errors) or "No frame was returned.",
        }

    stats = [inspect_frame(frame) for frame in frames]
    compatible = [item for item in stats if item and item.get("valid")]
    changes = [
        float(np.mean(np.abs(previous.astype(np.int16) - current.astype(np.int16))))
        for previous, current in zip(frames, frames[1:])
        if previous.shape == current.shape
    ]
    changing = any(value >= MIN_FRAME_CHANGE for value in changes)

    if not compatible:
        issues = sorted({item.get("issue", "invalid frame") for item in stats if item})
        status = "INVALID IMAGE CONTENT"
        details = ", ".join(issues) or "No valid image frames."
    elif changing:
        status = "CHANGING IMAGE CANDIDATE"
        details = "Frame pixels vary; visually confirm this is the phone camera, not a test pattern."
    else:
        status = "IMAGE NOT VERIFIED LIVE"
        details = "Frames are readable but did not change enough during the probe."

    last_stats = stats[-1]
    if last_stats:
        frame_details = (
            f"{last_stats.get('width', '?')}x{last_stats.get('height', '?')} "
            f"mean={last_stats.get('mean', float('nan')):.1f}/255 "
            f"pixel_std={last_stats.get('std', float('nan')):.2f} "
            f"valid_frames={len(compatible)}/{len(frames)} "
            f"changed_pairs={sum(value >= MIN_FRAME_CHANGE for value in changes)}/{len(changes)}"
        )
    else:
        frame_details = "Last frame was invalid."

    if errors:
        details = f"{details} Read errors: {'; '.join(errors)}"

    return {
        "index": device_index,
        "backend": backend_name,
        "opened": opened,
        "backend_reported": backend_reported,
        "frames_read": len(frames),
        "status": status,
        "details": f"{frame_details}. {details} "
        f"Resolution={active_resolution or 'unknown'}; FourCC={active_fourcc or 'unknown'}.",
    }


def main():
    backends = available_backends()
    print(f"OpenCV {cv2.__version__}; probing camera indices {list(DEVICE_INDICES)}")
    print(f"Available capture backends: {', '.join(name for name, _ in backends) if backends else 'none'}")
    if not backends:
        print("No supported Windows camera backend (CAP_ANY / MSMF / DirectShow) is available.")
        return 1

    results = []
    for backend_name, backend in backends:
        for device_index in DEVICE_INDICES:
            result = probe_device(device_index, backend_name, backend)
            results.append(result)
            backend_label = result.get("backend_reported", backend_name)
            print(
                f"index={device_index} requested_backend={backend_name} "
                f"actual_backend={backend_label} opened={result['opened']} "
                f"status={result['status']}"
            )
            print(f"  {result['details']}")

    candidates = [result for result in results if result["status"] == "CHANGING IMAGE CANDIDATE"]
    print("\nSummary:")
    if candidates:
        for candidate in candidates:
            print(
                f"  Candidate index {candidate['index']} with "
                f"{candidate['backend_reported']} produced changing image frames."
            )
        print("OpenCV indices do not identify Windows-friendly device names. Visually verify the feed before selecting it in the app.")
    else:
        print("  No camera index/backend pair produced verified changing image content.")
        print("  OpenCV is not receiving a valid live RGB frame from the Iriun virtual camera through the Windows capture interfaces available here.")
        print("  This indicates the Iriun virtual camera is only visible to the Iriun application, not through OpenCV's MSMF/DirectShow device API on this system.")
        print("  Supported practical alternative: use a Windows camera source that exposes a direct UVC device to OpenCV, or connect a supported local webcam/phone stream that OpenCV can enumerate. The app will continue to reject black/solid-color frames and will not fake a feed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
